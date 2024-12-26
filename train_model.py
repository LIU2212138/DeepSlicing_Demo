import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Master import CNN
# 数据加载器
def load_mnist_data(data_dir):
    """
    加载 MNIST 数据集。
    :param data_dir: 数据集文件夹路径
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])

    # 加载训练和测试数据
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    训练模型。
    :param model: 要训练的模型
    :param train_loader: 训练数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练的轮数
    """
    model.train()  # 模型进入训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 测试模型
def test_model(model, test_loader):
    """
    测试模型的准确率。
    :param model: 要测试的模型
    :param test_loader: 测试数据加载器
    """
    model.eval()  # 模型进入评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 保存权重
def save_model(model, save_path):
    """
    保存模型权重。
    :param model: 要保存的模型
    :param save_path: 保存路径
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

# 主函数
if __name__ == "__main__":
    # 数据路径
    data_dir = "./mnist_data"

    # 加载数据
    train_loader, test_loader = load_mnist_data(data_dir)

    # 初始化模型、损失函数和优化器
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=5)

    # 测试模型
    test_model(model, test_loader)

    # 保存模型权重
    save_model(model, "mnist_cnn.pth")
