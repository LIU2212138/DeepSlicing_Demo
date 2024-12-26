import os
import struct
import numpy as np
from PIL import Image

def load_mnist_images(file_path):
    """
    加载 MNIST 图像数据。
    :param file_path: MNIST 图像文件路径 (例如 "train-images-idx3-ubyte")
    :return: 一个 numpy 数组，形状为 (num_images, 28, 28)
    """
    with open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number {magic} for image file"
        image_data = np.fromfile(f, dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        return images

def load_mnist_labels(file_path):
    """
    加载 MNIST 标签数据。
    :param file_path: MNIST 标签文件路径 (例如 "train-labels-idx1-ubyte")
    :return: 一个 numpy 数组，包含每张图片的标签
    """
    with open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number {magic} for label file"
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def save_images(images, labels, output_dir, num_to_save=10):
    """
    将 MNIST 图片保存为 PNG 文件，用于可视化。
    :param images: 图片数组，形状为 (num_images, 28, 28)
    :param labels: 对应的标签数组
    :param output_dir: 输出目录
    :param num_to_save: 保存的图片数量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(min(num_to_save, len(images))):
        img = Image.fromarray(images[i])
        label = labels[i]
        file_name = os.path.join(output_dir, f"mnist_{i}_label_{label}.png")
        img.save(file_name)
        print(f"Saved: {file_name}")

if __name__ == "__main__":
    # 数据集文件路径
    images_file = "./mnist_data/MNIST/raw/train-images-idx3-ubyte"
    labels_file = "./mnist_data/MNIST/raw/train-labels-idx1-ubyte"
    output_dir = "./mnist_images"

    # 加载图像和标签
    images = load_mnist_images(images_file)
    labels = load_mnist_labels(labels_file)

    # 保存前 10 张图片用于检查
    save_images(images, labels, output_dir, num_to_save=10)
