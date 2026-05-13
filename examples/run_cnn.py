import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------- 只改这里 ---------------------
from models import CNN  # 替换 MLP 为 CNN
# ----------------------------------------------------
from utils import load_cifar10, batch_generator

# 配置训练设备，自动选择 CUDA（GPU）或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备:", device)


def train_cnn():
    """
    CNN 模型训练主函数
    包含数据加载、模型初始化、训练循环、评估与模型保存
    """
    # 加载 CIFAR-10 数据集，并将图像展平为一维向量
    (train_x, train_y), (test_x, test_y) = load_cifar10(flatten=True)
    print(f"数据集加载完成：训练集 {len(train_x)} 样本，测试集 {len(test_x)} 样本")
    print(f"输入维度：{train_x.shape[1]}（CIFAR-10 32×32×3=3072）")

    # --------------------- 只改这里 ---------------------
    model = CNN()  # 使用 CNN 模型
    # ----------------------------------------------------
    print(model)

    # 定义损失函数（使用 PyTorch 官方交叉熵损失）
    criterion = nn.CrossEntropyLoss()

    # 提取模型参数
    params = model.parameters()

    # 定义优化器（使用 PyTorch 官方 Adam）
    optimizer = optim.Adam(params, lr=0.001)

    # 训练超参数
    epo = 100         # CNN 不需要 100 轮
    batch_size = 64
    print("开始训练...")

    # 训练主循环
    for epoch in range(epo):
        train_loss = 0.0

        for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)

            # 前向传播
            logits = model.forward(batch_x)  # CNN 直接输出 logits，不用 softmax

            # 计算损失
            loss = criterion(logits, batch_y)
            train_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估
        avg_train_loss = train_loss / (len(train_x) / batch_size)
        train_acc = model.accuracy(train_x[:1000], train_y[:1000])
        test_acc = model.accuracy(test_x, test_y)

        print(f"Epoch [{epoch+1}/{epo}] | 训练损失: {avg_train_loss:.4f} | "
              f"训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

    # 保存权重
    torch.save(model.state_dict(), "cnn_weights.pth")
    print("训练完成，CNN 模型权重已保存！")


if __name__ == "__main__":
    train_cnn()