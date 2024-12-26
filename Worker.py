import pickle
import socket
import threading
import time

import torch
import torch.nn as nn
import atexit
from Util import separater

class DataStorage:
    def __init__(self):
        """
        初始化存储结构，使用字典存储特征图切片。
        Key: 层编号或切片编号
        Value: torch.Tensor（特征图切片）
        """
        self.storage = {}  # 存储特征图的字典

    def put(self, key, data):
        """
        存储特征图切片。
        :param key: 唯一标识符（例如层编号或切片编号）
        :param data: torch.Tensor，特征图切片
        """
        if not isinstance(data, torch.Tensor):
            raise ValueError("Data must be a torch.Tensor")
        self.storage[key] = data
        print(f"Stored data for key: {key}")

    def get(self, key):
        """
        获取特征图切片。
        :param key: 特征图切片的唯一标识符
        :return: torch.Tensor，返回存储的特征图切片
        """
        if key not in self.storage:
            raise KeyError(f"Data for key '{key}' not found.")
        return self.storage[key]

    def delete(self, key):
        """
        删除指定的特征图切片，释放内存。
        :param key: 特征图切片的唯一标识符
        """
        if key in self.storage:
            del self.storage[key]
            print(f"Deleted data for key: {key}")
        else:
            print(f"Key '{key}' not found. Nothing to delete.")

    def clear(self):
        """
        清空所有存储的数据。
        """
        self.storage.clear()
        print("All data cleared.")

    def keys(self):
        """
        获取当前存储的所有特征图切片的键。
        :return: 切片标识符列表
        """
        return list(self.storage.keys())

    def __repr__(self):
        """
        显示当前存储的特征图切片信息。
        """
        return f"DataStorage(keys={list(self.storage.keys())})"

class Worker:
    def __init__(self, host, port=12345, local_port=None):
        """初始化 Worker 类，创建任务队列和数据存储。"""
        self.data_storage = DataStorage()  # 数据存储
        self.computation_thread = None  # 计算线程
        self.listening_thread = None  # 数据服务器线程
        self.model_save_mode = "cache"
        # CNN模型，初始化的时候为空
        self.model = None
        self.struct = None
        self.model_cache = {}  # 用于存储模型权重的本地缓存
        self.cid = None
        self.host = host
        self.port = port
        self.local_port = local_port
        self.conn = None
        self.stop_flag = False
        self.split_dim = None
        self.jobs = {}
        atexit.register(self.stop)


    # 从对应网络端口获取模型
    def register(self, host, port):
        """
        从服务器获取模型并存入本地缓存
        """
        try:
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 如果指定了本地端口，绑定本地地址和端口
            if self.local_port is not None:
                self.conn.bind(("0.0.0.0", self.local_port))  # 绑定到所有本地 IP
            self.conn.connect((host, port))
            print(f"Connected to server {host}:{port}")
            self.start_listening()
            # 发送注册请求
            request = {"action": "register"}
            self.conn.sendall(pickle.dumps(request))
        except socket.error as e:
            print(f"Failed to connect to server: {e}")
            self.conn = None

    def start_listening(self):
        """
        启动消息监听线程
        """
        self.listening_thread = threading.Thread(target=self.listen_for_messages)
        self.listening_thread.start()


    def listen_for_messages(self):
        """
        持续监听服务器发送的消息（阻塞）
        """
        try:
            buffer = b""
            while not self.stop_flag:
                data = self.conn.recv(4096)  # 阻塞等待服务器消息
                if not data:
                    print("Connection closed by server.")
                    break
                buffer += data

                # 按分隔符拆分消息
                while separater in buffer:
                    msg, buffer = buffer.split(separater, 1)
                    try:
                        # 使用 pickle 反序列化消息
                        message = pickle.loads(msg)
                        print("Received structured message:", message)
                        self.process_message(message)  # 调用自定义消息处理逻辑
                    except pickle.UnpicklingError:
                        print("Failed to decode message with pickle. Skipping...")
        except socket.error as e:
            print(f"Error receiving message: {e}")
        finally:
            self.close_connection()

    def send_message(self, message):
        """
        主动向服务器发送消息
        :param message: 要发送的消息
        """
        if not self.conn:
            print("No active connection. Please connect to the server first.")
            return

        try:
            self.conn.sendall(pickle.dumps(message))
            print(f"Message sent: {message}")
        except socket.error as e:
            print(f"Failed to send message: {e}")

    def process_message(self, message):
        action = message["action"]
        if action == "register":
            self.cid = message["cid"]
            self.split_dim = message["split_dim"]
            model_data = message["data"]
            self.struct = model_data["structure"]
            self.model_cache = model_data["state_dict"]
            print("Model structure and weights cached successfully!")
        elif action == "job":
            job = message["job"]
            self.jobs[job["index"]] = job
            if job["finished"]:
                print(f"Job {job['index']} already finished.")
                response = {
                    "action": "job",
                    "index": job["index"],
                    "result": None,
                    "time": job["time"],
                    "finished": True,
                }
                self.send_message(response)
                return
            layers = job["layers"]
            feature_map = job["feature_map"]
            self.data_storage.put(job["index"], feature_map)
            input_range = job["range"]
            output_range = job["output_range"]
            start_time = time.time()
            self.assemble_model(layers)
            self.load_weights()
            # 输出feature_map尺寸
            print("Feature map shape:", feature_map.shape)
            output = self.infer(feature_map)
            end_time = time.time()

            # 输出output尺寸
            print("Output shape:", output.shape)
            # # 计算局部 output_range（映射到局部特征图范围）
            # local_output_start = output_range[0] - input_range[0]
            # local_output_end = output_range[1] - input_range[0]
            # local_output_range = (local_output_start, local_output_end)
            # # 根据局部 output_range 截取推理结果
            # if self.split_dim == "height":
            #     output = output[:, :, local_output_range[0]:local_output_range[1], :]  # 按高度截取
            # else:  # split_dim == "width"
            #     output = output[:, :, :, local_output_range[0]:local_output_range[1]]  # 按宽度截取

            print("Output shape after cropping:", output.shape)

            job["result"] = output
            job["finished"] = True
            job["time"] = end_time - start_time

            response = {
                "action": "job",
                "index": job["index"],
                "result": output,
                "time": job["time"],
                "finished": True,
            }
            self.send_message(response)
        pass

    def assemble_model(self, layer_names):
        """
        根据缓存的模型结构动态组装模型
        """
        if not self.struct:
            raise ValueError("No model structure found. Please register first!")

        if not layer_names:
            raise ValueError("Layer names must be provided.")

        # 根据需要组装的层名动态组装模型
        struct_needed = {k: v for k, v in self.struct.items() if k in layer_names}

        layers = {}
        for name, info in struct_needed.items():
            layer_type = info["type"]
            if layer_type == "Conv2d":
                params = info["params"]
                in_channels, out_channels, kernel_size = params["weight"][1], params["weight"][0], params["weight"][2:]
                padding = params["padding"]
                stride = params["stride"]
                dilation = params["dilation"]
                layers[name] = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation)
            elif layer_type == "MaxPool2d":
                params = info["params"]
                kernel_size = params["kernel_size"]
                padding = params["padding"]
                stride = params["stride"]
                dilation = params["dilation"]
                layers[name] = nn.MaxPool2d(kernel_size, padding=padding, stride=stride, dilation=dilation)
            elif layer_type == "ReLU":
                layers[name] = nn.ReLU()
            # 根据需要扩展支持的层类型

        # 动态组装模型
        self.model = nn.Sequential(*layers.values())
        print("Model assembled successfully!")

    def load_weights(self):
        """
        加载缓存的权重到动态组装的模型
        """
        if not self.model:
            raise ValueError("Model is not assembled yet. Please call assemble_model first.")

        self.model.load_state_dict(self.model_cache, strict=False)
        print("Model weights loaded successfully!")

    def infer(self, input_data):
        """
        使用动态组装的模型进行推理
        :param self: 包含组装和加载权重模型的实例
        :param input_data: 输入数据 (torch.Tensor)
        :return: 模型输出
        """
        if not self.model:
            raise ValueError("Model is not assembled yet. Please assemble the model first!")

        # 将模型切换到评估模式
        self.model.eval()

        # 禁用梯度计算（推理时无需计算梯度以节省内存）
        with torch.no_grad():
            output = self.model(input_data)

        return output

    def close_connection(self):
        """
        关闭客户端连接
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Connection closed.")

    def start(self):
        """
        启动客户端
        """
        self.register(self.host, self.port)

    def stop(self):
        """
        停止客户端
        """
        self.stop_flag = True
        self.close_connection()

if __name__ == "__main__":
    worker = Worker("127.0.0.1", 12345, 12347)
    worker.start()

