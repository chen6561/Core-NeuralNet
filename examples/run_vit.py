import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------- 只改这里 ---------------------
from models import ViT  # 替换为ViT模型
# ----------------------------------------------------
from utils import load_cifar10, batch_generator

# 配置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备:", device)


def train_vit():
    """
    ViT 模型训练主函数
    包含数据加载、模型初始化、训练循环、评估与模型保存
    """
    # 加载 CIFAR-10 数据集（展平为一维向量，兼容原数据加载逻辑）
    (train_x, train_y), (test_x, test_y) = load_cifar10(flatten=True)
    print(f"数据集加载完成：训练集 {len(train_x)} 样本，测试集 {len(test_x)} 样本")
    print(f"输入维度：{train_x.shape[1]}（CIFAR-10 32×32×3=3072）")

    # --------------------- 只改这里 ---------------------
    model = ViT(
        num_classes=10,
        img_size=32,
        patch_size=4,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        hidden_dim=512,
        dropout=0.1
    )  # 初始化ViT模型
    # ----------------------------------------------------
    print(model)

    # 损失函数（交叉熵）
    criterion = nn.CrossEntropyLoss()

    # 优化器（Adam，适配ViT的学习率）
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)

    # 训练超参数（ViT收敛更快，减少训练轮数）
    epo = 150
    batch_size = 128
    print("开始训练ViT...")

    # 训练主循环
    for epoch in range(epo):
        train_loss = 0.0

        for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)

            # 前向传播
            logits = model.forward(batch_x)

            # 计算损失
            loss = criterion(logits, batch_y)
            train_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估（训练集取前1000样本加速）
        avg_train_loss = train_loss / (len(train_x) / batch_size)
        train_acc = model.accuracy(train_x[:1000], train_y[:1000])
        test_acc = model.accuracy(test_x, test_y)

        print(f"Epoch [{epoch+1}/{epo}] | 训练损失: {avg_train_loss:.4f} | "
              f"训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

    # 保存权重
    torch.save(model.state_dict(), "vit_weights.pth")
    print("训练完成，ViT 模型权重已保存！")


if __name__ == "__main__":
    train_vit()