import math
import queue
import threading
import time
import uuid
from itertools import batched
from PIL import Image
import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import socket
import pickle
import atexit
from Util import separater
import numpy as np

def cleanup_socket(sock):
    sock.close()
    print("Socket closed at exit.")

# 定义一个简单模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道为1，输出通道为32
        self.relu1 = nn.ReLU()  # 激活函数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，大小为2x2，步幅为2

        # 第二层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道为32，输出通道为64
        self.relu2 = nn.ReLU()  # 激活函数

        # 全连接层
        self.fc = nn.Linear(64 * 14 * 14, 10)  # 将特征图展平后输入，输出10个分类（以MNIST为例）

    def forward(self, x):
        # 第一层卷积 -> 激活 -> 池化
        x = self.relu1(self.conv1(x))
        x = self.pool(x)

        # 第二层卷积 -> 激活
        x = self.relu2(self.conv2(x))

        # 展平特征图（batch_size, channels * height * width）
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        return x

class ClientInfo:
    def __init__(self, cid, addr, conn):
        self.cid = cid
        self.addr = addr
        self.connection = conn
    def get_addr(self):
        return self.addr
    def get_cid(self):
        return self.cid
    def get_conn(self):
        return self.connection

class Master:
    def __init__(self, model, port=12345, client_num=1, block_num=1, oriData=None):
        if oriData is None:
            oriData = []
        self.model = model
        self.port = port
        self.oriData = queue.Queue()
        self.oriData.put(self.load_single_image_as_tensor("./mnist_images/mnist_8_label_1.png"))
        self.split_dim = None
        self.output_ranges = {}
        self.fc = None
        self.struct, self.runtime_info = self.parse_model_structure()
        self.new_conn_stop_flag = False
        self.clients = {}
        self.client_num = client_num
        self.conn_lock = threading.Lock()
        self.process_stop_flag = False
        self.listen_for_connection_thread = None
        self.process_connections_thread = None
        self.block_num = block_num
        self.blocks = {}
        self.cut_blocks()
        self.job_batchs = {}

    # 模型结构解析器
    def parse_model_structure(self):
        """
        解析模型结构并计算每一层的特征图输入和输出尺寸。
        """
        if self.oriData.empty():
            raise ValueError("The oriData queue is empty. Please provide input data.")

        # 从 oriData 中获取一个输入样本，并提取输入特征图形状
        sample_data = self.oriData.queue[0]  # 假设数据是形状为 (batch_size, channels, height, width)
        _, channels, height, width = sample_data.shape
        if height >= width: # 谁大谁就是split_dim
            self.split_dim = "height"
        else:
            self.split_dim = "width"
        input_shape = (channels, height, width)  # 初始输入特征图形状

        structure = {}
        info = {}
        pre_layer = None
        index = 0
        current_shape = input_shape  # 当前特征图尺寸

        for name, module in self.model.named_modules():
            # fc层存起来，最后在master中执行
            if "fc" in name:
                self.fc = module
                continue
            if name:  # 跳过根模块
                # 提取层的参数
                layer_type = type(module).__name__
                layer_params = {k: v.shape for k, v in module.state_dict().items()}

                # 初始化结构
                structure[name] = {
                    "type": layer_type,
                    "params": layer_params
                }

                # 计算当前层的输出特征图尺寸
                if layer_type == "Conv2d":
                    # 获取卷积层参数
                    conv_params = module
                    kernel_size = conv_params.kernel_size
                    stride = conv_params.stride
                    padding = conv_params.padding
                    out_channels = conv_params.out_channels
                    dilation = conv_params.dilation
                    # 全部添加到结构中
                    if self.split_dim == "height":
                        structure[name]["params"]["kernel_size"] = kernel_size[0]
                        structure[name]["params"]["stride"] = stride[0]
                        structure[name]["params"]["padding"] = padding[0]
                        structure[name]["params"]["dilation"] = dilation
                    else:
                        structure[name]["params"]["kernel_size"] = kernel_size[1]
                        structure[name]["params"]["stride"] = stride[1]
                        structure[name]["params"]["padding"] = padding[1]
                        structure[name]["params"]["dilation"] = dilation
                    structure[name]["params"]["out_channels"] = out_channels

                    # 计算输出特征图尺寸
                    current_height = (current_shape[1] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
                    current_width = (current_shape[2] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
                    current_shape = (out_channels, current_height, current_width)

                elif layer_type == "MaxPool2d":
                    # 获取池化层参数
                    pool_params = module
                    kernel_size = pool_params.kernel_size
                    stride = pool_params.stride
                    padding = pool_params.padding
                    dilation = pool_params.dilation
                    # 全部添加到结构中
                    structure[name]["params"]["kernel_size"] = kernel_size
                    structure[name]["params"]["stride"] = stride
                    structure[name]["params"]["padding"] = padding
                    structure[name]["params"]["dilation"] = dilation

                    # 计算输出特征图尺寸
                    current_height = (current_shape[1] + 2 * padding - kernel_size) // stride + 1
                    current_width = (current_shape[2] + 2 * padding - kernel_size) // stride + 1
                    current_shape = (current_shape[0], current_height, current_width)

                elif layer_type == "ReLU":
                    # ReLU 不改变特征图尺寸
                    pass

                # 初始化信息
                info[name] = {
                    "index": index,
                    "type": layer_type,
                    "input_shape": input_shape,
                    "output_shape": current_shape
                }
                if self.split_dim == "height":
                    self.output_ranges[name] = current_shape[1]
                else:
                    self.output_ranges[name] = current_shape[2]

                # 记录上一层的链接关系
                if pre_layer is not None:
                    info[pre_layer]["next_layer"] = name
                    info[name]["previous_layer"] = [pre_layer]

                # 更新下一层的输入形状
                input_shape = current_shape
                pre_layer = name
                index += 1

        return structure, info

    def listen_for_connection(self, port):
        # 解析模型结构
        structure = self.struct
        # 获取模型权重
        state_dict = self.model.state_dict()

        # 打包数据
        data = {
            "structure": structure,
            "state_dict": state_dict
        }

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            s.listen(self.client_num)

            atexit.register(cleanup_socket, s)
            print(f"Master listening on port {port}...")
            while not self.new_conn_stop_flag:
                conn, addr = s.accept()

                print("Connected by", addr)
                # 接收请求
                dataRecv = conn.recv(1024)
                if not dataRecv:
                    continue
                request = pickle.loads(dataRecv)
                if request["action"] == "register":
                    if self.client_num == len(self.clients):
                        continue
                    cid = len(self.clients)
                    self.clients[cid] = ClientInfo(cid, addr, conn)
                    conn.sendall(pickle.dumps({"action": "register",
                                               "cid": cid,
                                               "data": data,
                                               "split_dim": self.split_dim,
                                               }) + separater)
                    if self.client_num == len(self.clients):
                        self.new_conn_stop_flag = True
                        for client in self.clients.values():
                            job = self.get_job(0, client)
                            self.assign_job(job, client.get_conn())

    def send_data(self, conn, data):
        with conn.makefile('wb') as f:
            pickle.dump(data, f)

    def receive_data_with_select(self, conn, timeout=0.1):
        """
        使用 select 检查套接字是否可读
        :param conn: 非阻塞模式的 socket 对象
        :param timeout: 超时时间
        :return: 反序列化后的数据，或者 None
        """
        readable, _, _ = select.select([conn], [], [], timeout)
        if readable:
            try:
                with conn.makefile('rb') as f:
                    return pickle.load(f)  # 读取并反序列化数据
            except EOFError:
                # 数据流结束
                print("Connection closed by peer.")
                return None
            except pickle.UnpicklingError as e:
                print(f"Failed to decode message: {e}")
                return None
        else:
            # 没有数据可读
            print("No data received within the timeout period.")
            return None

    def process_connections(self):
        """
        轮询处理连接中的消息，使用分割符分割消息。
        """
        # buffer = {}  # 为每个客户端维护独立的接收缓冲区

        while not self.process_stop_flag:
            for client_info in list(self.clients.values()):
                conn = client_info.get_conn()
                addr = client_info.get_addr()
                try:
                    received = self.receive_data_with_select(conn)
                    if received is not None:
                        self.process_request(received, conn, client_info)
                except socket.error as e:
                    print(f"Error processing connection with {addr}: {e}")
                    conn.close()
                    del self.clients[addr]
                    continue

                # if addr not in buffer:
                #     buffer[addr] = b""  # 初始化缓冲区
                #
                # try:
                #     # 接收数据
                #     data = conn.recv(4096)
                #     if not data:
                #         # 如果接收的数据为空，说明连接已关闭
                #         print(f"Connection with {addr} closed by client.")
                #         conn.close()
                #         del self.clients[addr]
                #         del buffer[addr]
                #         continue
                #
                #     # 将数据加入缓冲区
                #     buffer[addr] += data
                #
                #     # 按分割符拆分消息
                #     while separater in buffer[addr]:
                #         message_data, buffer[addr] = buffer[addr].split(separater, 1)
                #         try:
                #             # 解码消息
                #             request = pickle.loads(message_data)
                #             self.process_request(request, conn, client_info)
                #         except pickle.UnpicklingError as e:
                #             print(f"Failed to decode message from {addr}: {e}")
                #             continue
                #
                # except BlockingIOError:
                #     # 如果没有数据可读，跳过该连接
                #     continue
                # except socket.error as e:
                #     print(f"Error processing connection with {addr}: {e}")
                #     conn.close()
                #     del self.clients[addr]
                #     del buffer[addr]
                time.sleep(0.1)  # 防止轮询线程占用过多 CPU

    def process_request(self, request, conn, client_info):
        if request["action"] == "garbage":
            pass
        elif request["action"] == "job":
            index = request["index"]
            self.job_batchs[index][client_info.get_cid()]["result"] = request["result"]
            #输出resukt尺寸
            print("Result shape:", request["result"].shape)
            self.job_batchs[index][client_info.get_cid()]["finished"] = True
            self.job_batchs[index][client_info.get_cid()]["time"] = request["time"]
            if all([job["finished"] for job in self.job_batchs[index].values()]):
                if index == len(self.blocks) - 1:
                    if self.fc is not None:
                        feature_map = self.assemble_feature_map(index)
                        # 输出尺寸
                        print("Feature map shape:", feature_map.shape)
                        output = self.infer_with_fc(feature_map)
                        print("Inference result:", output)
                        self.display_prediction(output)
                    print("All jobs finished.")
                    return
                for client in self.clients.values():
                    job = self.get_job(index + 1, client)
                    self.assign_job(job, client.get_conn())
        elif request["action"] == "task":
            pass
        pass

    def cut_blocks(self):
        total_len = len(self.runtime_info)
        cut_point_list = [i for i in range(0, total_len, total_len // self.block_num)]  # 还要移除0
        cut_point_list = cut_point_list[1:]
        current_block = 0
        for i, (layer_name, layer_info) in enumerate(self.runtime_info.items()):
            if self.blocks.get(current_block) is None:
                self.blocks[current_block] = []
            self.blocks[current_block].append(layer_name)
            if i in cut_point_list:
                current_block += 1

    def assemble_feature_map(self, index):
        # 将self.job_batchs[index][client.get_cid()]["range"]的feature_map拼接起来
        feature_map = None
        for client in self.clients.values():
            if self.job_batchs[index][client.get_cid()]["result"] is None:
                continue
            if feature_map is None:
                feature_map = self.job_batchs[index][client.get_cid()]["result"]
            else:
                # 根据self.split_dim进行拼接
                if self.split_dim == "height":
                    #从height维度拼接
                    feature_map = torch.cat((feature_map, self.job_batchs[index][client.get_cid()]["result"]), dim=2)
                else:
                    feature_map = torch.cat((feature_map, self.job_batchs[index][client.get_cid()]["result"]), dim=3)
        return feature_map

    def infer_with_fc(self, feature_map):
        """
        使用存储的全连接层 (fc) 对特征图进行推理
        :param feature_map: 输入的特征图 (torch.Tensor)
        :return: 全连接层的输出
        """
        if not hasattr(self, 'fc') or self.fc is None:
            raise ValueError("No fully connected layer (fc) found. Ensure the model contains an fc layer.")

        # 将特征图展平为 (batch_size, -1) 的形状，作为全连接层的输入
        flattened_input = feature_map.view(feature_map.size(0), -1)  # 保证 batch_size 维度正确

        # 使用存储的 fc 层进行推理
        output = self.fc(flattened_input)

        return output

    def generate_job_batch(self, index):
        block = self.blocks[index]
        last_layer = block[-1]
        last_layer_range = self.runtime_info[last_layer]["output_shape"]
        self.job_batchs[index] = {}
        data = None
        if index == 0:
            data = self.oriData.get()
            for client in self.clients.values():
                # 这就是单个任务的数据
                self.job_batchs[index][client.get_cid()] = {}
                self.job_batchs[index][client.get_cid()]["index"] = index
                self.job_batchs[index][client.get_cid()]["layers"] = block
                self.job_batchs[index][client.get_cid()]["output_range"] = None

                # 平均分配最后一层的输出
                if self.split_dim == "height":
                    client_id = client.get_cid()
                    current_last_layer_range = [last_layer_range[1] // self.client_num * client_id,
                                                min(last_layer_range[1] // self.client_num * (client_id + 1),
                                                    last_layer_range[1])]
                    self.job_batchs[index][client.get_cid()]["range"], _ = self.AIR(block[0], last_layer,
                                                                                    current_last_layer_range)
                    self.job_batchs[index][client.get_cid()]["output_range"] = current_last_layer_range
                    # 根据高度切分
                    self.job_batchs[index][client.get_cid()]["feature_map"] = data[:, :, self.job_batchs[index][client.get_cid()][
                                                                                  "range"][0]:
                                                                              self.job_batchs[index][client.get_cid()][
                                                                                  "range"][1],:]
                else:
                    client_id = client.get_cid()
                    current_last_layer_range = [last_layer_range[2] // self.client_num * client_id,
                                                min(last_layer_range[2] // self.client_num * (client_id + 1),
                                                    last_layer_range[2])]
                    self.job_batchs[index][client.get_cid()]["range"], _ = self.AIR(block[0], last_layer,
                                                                                    current_last_layer_range)
                    self.job_batchs[index][client.get_cid()]["output_range"] = current_last_layer_range
                    self.job_batchs[index][client.get_cid()]["feature_map"] = data[:, :, :, self.job_batchs[index][client.get_cid()][
                                                                                  "range"][0]:
                                                                              self.job_batchs[index][client.get_cid()][
                                                                                  "range"][1]]

                self.job_batchs[index][client.get_cid()]["finished"] = False
                self.job_batchs[index][client.get_cid()]["result"] = None
                self.job_batchs[index][client.get_cid()]["time"] = None
        else:
            data = self.assemble_feature_map(index - 1)
            for client in self.clients.values():
                # 这就是单个任务的数据
                self.job_batchs[index][client.get_cid()] = {}
                self.job_batchs[index][client.get_cid()]["index"] = index
                self.job_batchs[index][client.get_cid()]["layers"] = block
                self.job_batchs[index][client.get_cid()]["output_range"] = None
                self.job_batchs[index][client.get_cid()]["finished"] = False
                self.job_batchs[index][client.get_cid()]["result"] = None
                self.job_batchs[index][client.get_cid()]["time"] = None

            Pc = self.blocks[index - 1][-1]
            Pc1 = last_layer
            rw = [(client_data["output_range"][1] - client_data["output_range"][0]) for client_data in
                  self.job_batchs[index - 1].values() if
                  "output_range" in client_data]
            tw = [client_data["time"] for client_data in self.job_batchs[index - 1].values() if
                  "time" in client_data]
            layers = block
            workers = self.clients
            self.PSS(Pc, Pc1, rw, tw, layers, workers, self.output_ranges, self.job_batchs[index])

            print("data shape:", data.shape)
            for client in self.clients.values():
                print("client range:", self.job_batchs[index][client.get_cid()]["range"])
                if self.split_dim == "height":
                    self.job_batchs[index][client.get_cid()]["feature_map"] = data[:, :, self.job_batchs[index][client.get_cid()][
                                                                                  "range"][0]:
                                                                              self.job_batchs[index][client.get_cid()][
                                                                                  "range"][1],:]
                    print("feature map shape:", self.job_batchs[index][client.get_cid()]["feature_map"].shape)
                    print("expected output range:", self.job_batchs[index][client.get_cid()]["output_range"])
                    print("-------------------------------------------------------------")
                else:
                    self.job_batchs[index][client.get_cid()]["feature_map"] = data[:, :,
                                                                              :, self.job_batchs[index][client.get_cid()][
                                                                                  "range"][0]:
                                                                              self.job_batchs[index][client.get_cid()][
                                                                                  "range"][1]]


    def allocate_range_with_minimum(self, total, ratios, min_allocation=1):
        """
        根据给定比例分配整数区间，确保总和为 total，且所有比例对应区间均至少分配到 min_allocation。
        :param total: 总区间长度（整数）
        :param ratios: 比例列表（浮点数，总和为 1）
        :param min_allocation: 每个分配的最小值（整数）
        :return: 分配结果列表，每个元素是对应区间的宽度
        """
        import math

        # 确保总区间长度足够满足最小分配要求
        n = len(ratios)
        if total < n * min_allocation:
            raise ValueError("Total range is too small to satisfy the minimum allocation requirement.")

        # 步骤 1: 为每个区间分配最小值
        allocations = [min_allocation] * n
        remaining = total - sum(allocations)  # 计算剩余可分配长度

        # 步骤 2: 根据比例计算初步分配
        float_allocations = [ratio * remaining for ratio in ratios]

        # 步骤 3: 向下取整分配
        int_allocations = [math.floor(alloc) for alloc in float_allocations]

        # 步骤 4: 计算剩余宽度
        new_remaining = remaining - sum(int_allocations)

        # 步骤 5: 根据比例的浮点数误差（小数部分）排序
        decimal_parts = [(float_allocations[i] - int_allocations[i], i) for i in range(len(ratios))]
        decimal_parts.sort(reverse=True, key=lambda x: x[0])  # 按小数部分降序排列

        # 分配剩余的宽度
        for _, idx in decimal_parts[:new_remaining]:
            int_allocations[idx] += 1

        # 将分配结果加上最小值
        final_allocations = [int_allocations[i] + min_allocation for i in range(n)]

        return final_allocations

    def PSS(self, Pc, Pc1, rw, tw, layers, workers, output_ranges, job_batches):
        """
        实现 PSS 算法，生成新的作业任务。
        :param max_kernel_size:
        :param Pc1:
        :param Pc: 当前的同步点
        :param rw: 每个 worker 的输入范围
        :param tw: 每个 worker 最后一个任务的时间成本
        :param layers: 当前的模型层信息
        :param workers: 当前所有 worker 的列表
        :param output_ranges: 每一层的输出范围信息 (ξL)
        :param job_batches: 每个客户端的当前作业批次（直接更新，不需要返回）
        """
        W = len(workers)  # Worker 总数

        # Step 1: 计算比例 sw
        sw = [max(tw[i], 1e-4) / rw[i] for i in range(W)]

        kernel_sizes = {name: layer["params"]["kernel_size"]
            for name, layer in self.struct.items()
            if "params" in layer and "kernel_size" in layer["params"]}

        exp_range, S = self.AIR(Pc, Pc1, [0, output_ranges[Pc1]])
        for layer in layers:
            if layer not in kernel_sizes:
                continue
            exp_input_length_for_layer = S[layer][0][1] - S[layer][0][0]
            if exp_input_length_for_layer < kernel_sizes[layer] * W:
                print(f"Expected input length for layer {layer} is too small to allocate.")
                # 直接分配给一个tw最小的worker
                min_tw = min(tw)
                min_tw_index = tw.index(min_tw)
                for w in range(W):
                    job_batches[w]["range"] = [0, 0]
                    job_batches[w]["output_range"] = [0, 0]
                    job_batches[w]["finished"] = True
                    # 取平均
                    job_batches[w]["time"] = sum(tw) / W
                job_batches[min_tw_index]["range"] = exp_range
                job_batches[min_tw_index]["output_range"] = output_ranges[Pc1]


        # Step 2: 分配范围
        # Step 2: 初步划分 Jw
        ratios = []
        for w in range(W):
            # 使用比例 sw 划分任务范围(int向上取整)
            ratios.append(sw[w] / sum(sw))
        lengths = self.allocate_range_with_minimum(output_ranges[Pc1], ratios, min_allocation=3)
        Jw = []
        b = 0
        for w in range(W):
            a = b
            b = a + lengths[w]
            Jw.append((a, b))

        # Step 3: 迭代优化范围
        new_Jw = Jw
        for _ in range(1000):
            tau = [None] * W
            new_tau = [None] * W
            new_input_ranges = []
            for w in range(W):
                # 调用 AIR 算法计算输入范围
                expected_out_range = Jw[w]  # 当前 Worker 的输出范围
                input_range, S = self.AIR(Pc, Pc1, expected_out_range)
                xin, yin = input_range
                # 更新 job_batches 的输入范围
                job_batches[w]["range"] = (xin, yin)
                # 计算时间成本
                tau[w] = (yin - xin) / sw[w]

                # 新的比例
                new_exp_range = new_Jw[w]
                new_input_range, _ = self.AIR(Pc, Pc1, new_exp_range)
                new_input_ranges.append(new_input_range)
                new_tau[w] = (new_input_range[1] - new_input_range[0]) / sw[w]

            # 检查负载均衡
            if max(tau) - min(tau) > max(new_tau) - min(new_tau):
                for w in range(W):
                    Jw[w] = new_Jw[w]
                    job_batches[w]["range"] = new_input_ranges[w]
            # 调整范围
            new_Jw = self.fine_tune(Jw, output_ranges[Pc1])  # 调用 fine_tune 进行范围调整

        # Step 4: 更新 job_batches 的 output_range
        for w in range(W):
            # 使用最终输入范围 Jw[w]，直接更新 output_range
            job_batches[w]["output_range"] = Jw[w]

    def fine_tune(self, Jw, output_range):
        """
        调整 Worker 的任务范围以实现负载均衡。
        :param Jw_w: 当前 Worker 的任务范围 [start, end]
        :param tau: 每个 Worker 的计算时间成本列表
        :param w: 当前 Worker 的索引
        :return: 调整后的任务范围
        """
        # 随机一个和为1，长度为W的数组
        W = len(Jw)
        # 随机生成一个比例数组
        ratios = self.generate_random_ratios(W)
        # 计算新的范围
        lengths = self.allocate_range_with_minimum(output_range, ratios, min_allocation=3)
        Jw = []
        b = 0
        for w in range(W):
            a = b
            b = a + lengths[w]
            Jw.append((a, b))
        return Jw



    def generate_random_ratios(self, W):
        """
        随机生成一个比例数组，长度为 W，总和为 1
        :param W: 数组长度
        :return: 随机比例数组
        """
        ratios = np.random.rand(W)  # 生成 W 个随机数
        ratios /= ratios.sum()  # 将每个数除以总和，归一化为比例
        return ratios

    def AIR(self, input_layer, output_layer, expected_out_range, S=None):
        """
        AIR 算法实现: 计算满足输出范围的输入范围，并推导中间层的输出范围。

        :param input_layer: 输入层名称（str）
        :param output_layer: 输出层名称（str）
        :param expected_out_range: 期望的输出范围（[x_out, y_out]）
        :param S: 存储中间层范围的字典，用于记忆化优化（递归共享）
        :return: (输入范围, 中间层输出范围集合 S)
        """
        # 初始化存储中间层范围的集合 S（第一次调用时）
        if S is None:
            S = {}

        # 如果 S 中已包含当前输出范围，直接返回
        if output_layer in S:
            if expected_out_range in S[output_layer]:
                return None, S  # 无需更新
            else:
                # 合并范围
                S[output_layer].append(expected_out_range)
        else:
            S[output_layer] = [expected_out_range]

        # 获取当前层的信息
        output_layer_info = self.runtime_info[output_layer]

        # 如果输入层和输出层是同一层，直接返回
        if input_layer == output_layer:
            input_range = self.calculate_input_range(output_layer, expected_out_range)
            return input_range, S

        # 如果当前层没有上游依赖层（无依赖层）
        if "previous_layer" not in output_layer_info or not output_layer_info["previous_layer"]:
            return None, S  # 无法推导范围

        # 初始化输入范围
        input_range = None

        # 遍历当前层的所有上游层（依赖层）
        for previous_layer in output_layer_info["previous_layer"]:
            # 计算当前层的输入范围
            prev_range = self.calculate_input_range(output_layer, expected_out_range)

            # 递归调用 AIR 计算上游层的输入范围
            prev_input_range, S = self.AIR(input_layer, previous_layer, prev_range, S)

            # 合并输入范围
            if prev_input_range is not None:
                if input_range is None:
                    input_range = prev_input_range
                else:
                    input_range = [
                        min(input_range[0], prev_input_range[0]),
                        max(input_range[1], prev_input_range[1]),
                    ]

        # 返回输入范围和中间层范围集合
        return input_range, S

    def calculate_input_range(self, layer_name, output_range):
        """
        根据输出范围计算输入范围（假设为反向传播操作）。
        :param layer_name: 当前层的名称
        :param output_range: 输出范围 [x_out, y_out]
        :return: 输入范围 [x_in, y_in]
        """
        # 获取层的信息
        layer_info = self.struct[layer_name]
        layer_type = layer_info["type"]

        if layer_type == "Conv2d":
            # 卷积层反向计算输入范围
            kernel_size = layer_info["params"]["kernel_size"]
            stride = layer_info["params"]["stride"]
            padding = layer_info["params"]["padding"]

            input_min = (output_range[0] - kernel_size + 1 + 2 * padding) // stride
            input_max = (output_range[1] - kernel_size + 1 + 2 * padding) // stride
            return [input_min, input_max]

        elif layer_type == "MaxPool2d":
            # MaxPool2d 反向计算输入范围
            kernel_size = layer_info["params"]["kernel_size"]
            stride = layer_info["params"]["stride"]
            padding = layer_info["params"]["padding"]

            input_min = stride * output_range[0] - padding
            input_max = stride * (output_range[1] - 1) - padding + kernel_size
            return [input_min, input_max]

        elif layer_type == "ReLU":
            # ReLU 层不会改变范围
            return output_range

        # 可扩展其他层类型
        else:
            raise NotImplementedError(f"Layer type {layer_type} is not supported.")

    def get_job(self, index, client_info):
        if self.job_batchs.get(index) is None:
            self.generate_job_batch(index)
        job = self.job_batchs[index][client_info.get_cid()]
        return job

    def assign_job(self, job, conn):
        conn.sendall(pickle.dumps({
            "action": "job",
            "job": job
        }) + separater)

    def start(self):
        self.listen_for_connection_thread = threading.Thread(target=self.listen_for_connection, args=(self.port,))
        self.process_connections_thread = threading.Thread(target=self.process_connections)
        self.listen_for_connection_thread.start()
        self.process_connections_thread.start()

        # 按q停止线程并退出
        while True:
            if input("Press 'q' to quit: ") == "q":
                self.process_stop_flag = True
                self.new_conn_stop_flag = True
                break

        self.listen_for_connection_thread.join()
        self.process_connections_thread.join()

    def load_single_image_as_tensor(self, image_path):
        """
        加载单张图片，并将其转换为 28x28 的 torch.Tensor 格式。
        :param image_path: 图片的路径
        :return: torch.Tensor，形状为 (1, 1, 28, 28)
        """
        # 打开图片并转换为灰度图（1通道）
        img = Image.open(image_path).convert("L")

        # 调整图片大小为 28x28（如果图片已经是 28x28，则此步骤可以省略）
        img = img.resize((28, 28))

        # 将图片转换为 numpy 数组
        img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化到 [0, 1]

        # 转换为 PyTorch 张量，形状为 (1, 1, 28, 28)
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

        return img_tensor

    def display_prediction(self, output):
        """
        将网络的输出转换为概率分布，并显示最可能的类别及其概率。
        :param output: 网络的原始输出 logits (torch.Tensor)
        """
        # 1. 使用 Softmax 转换为概率分布
        probabilities = F.softmax(output, dim=1)

        # 2. 获取最可能的类别及其概率
        predicted_class = torch.argmax(probabilities, dim=1).item()  # 最大概率对应的类别
        predicted_probability = probabilities[0, predicted_class].item()  # 最大概率值

        # 3. 打印结果
        print("Probabilities for each class:")
        for i, prob in enumerate(probabilities[0]):
            print(f"Class {i}: {prob:.4f}")

        print(f"\nPredicted class: {predicted_class} with probability: {predicted_probability:.4f}")

if __name__ == "__main__":
    # 初始化模型和 Master
    model = CNN()
    # 加载权重
    model.load_state_dict(torch.load("mnist_cnn.pth"))
    master = Master(model, client_num=2, block_num=5)

    print("Model structure:")
    print(master.struct)

    master.start()

