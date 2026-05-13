import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 导入自定义模型与工具函数
from models import MLP
from utils import load_cifar10, batch_generator

# 配置训练设备，自动选择 CUDA（GPU）或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备:", device)


def train_mlp():
    """
    MLP 模型训练主函数
    包含数据加载、模型初始化、训练循环、评估与模型保存
    """
    # 加载 CIFAR-10 数据集，并将图像展平为一维向量
    (train_x, train_y), (test_x, test_y) = load_cifar10(flatten=True)
    print(f"数据集加载完成：训练集 {len(train_x)} 样本，测试集 {len(test_x)} 样本")
    print(f"输入维度：{train_x.shape[1]}（CIFAR-10 32×32×3=3072）")

    # 模型超参数配置
    input_dim = 3072    # 输入特征维度（展平后）
    hidden_dims = [256, 128]  # 隐藏层维度
    output_dim = 10    # 分类类别数（CIFAR-10）

    # 初始化 MLP 模型并打印结构
    model = MLP(input_dim, hidden_dims, output_dim)
    print(model)

    # 定义损失函数（使用 PyTorch 官方交叉熵损失）
    criterion = nn.CrossEntropyLoss()

    # 提取模型中所有可训练参数（权重 + 偏置）
    params = []
    for layer in model.layers:
        params.append(layer.weights)
        params.append(layer.biases)

    # 定义优化器（使用 PyTorch 官方 Adam）
    optimizer = optim.Adam(params, lr=0.001)

    # 训练超参数
    epochs = 100        # 训练轮数
    batch_size = 64     # 批次大小
    print("开始训练...")

    # 训练主循环
    for epoch in range(epochs):
        train_loss = 0.0  # 累计训练损失

        # 批次训练
        for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
            # 将数据转换为 Tensor 并迁移到指定设备
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)

            # 前向传播：模型预测
            y_pred = model.forward(batch_x)
            # 将概率转换为 logits，适配 CrossEntropyLoss
            logits = torch.log(y_pred + 1e-7)

            # 计算损失
            loss = criterion(logits, batch_y)
            train_loss += loss.item()

            # 反向传播与参数更新
            optimizer.zero_grad()   # 清空梯度
            loss.backward()         # 反向传播计算梯度
            optimizer.step()        # 更新参数

        # 计算平均训练损失
        avg_train_loss = train_loss / (len(train_x) / batch_size)

        # 在训练集子集与测试集上计算准确率
        train_acc = model.accuracy(train_x[:1000], train_y[:1000])
        test_acc = model.accuracy(test_x, test_y)

        # 打印训练信息
        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.4f} | "
              f"训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

    # 保存模型权重与偏置（numpy 格式）
    weights_dict = {}
    biases_dict = {}
    for i, layer in enumerate(model.layers):
        weights_dict[f"layer_{i}_weights"] = layer.weights.detach().cpu().numpy()
        biases_dict[f"layer_{i}_biases"] = layer.bias.detach().cpu().numpy()

    np.savez("mlp_weights.npz", **weights_dict)
    np.savez("mlp_biases.npz", **biases_dict)
    print("训练完成，模型权重已保存！")


if __name__ == "__main__":
    train_mlp()