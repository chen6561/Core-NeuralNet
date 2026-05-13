import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 你原有模型完全不动！！！
from models import MLP
from utils import load_cifar10, batch_generator

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练设备:", device)

def train_mlp():
    (train_x, train_y), (test_x, test_y) = load_cifar10(flatten=True)
    print(f"数据集加载完成：训练集 {len(train_x)} 样本，测试集 {len(test_y)} 样本")
    print(f"输入维度：{train_x.shape[1]}（CIFAR-10 32×32×3=3072）")

    # 模型完全不动
    input_dim = 3072
    hidden_dims = [128, 64]
    output_dim = 10
    model = MLP(input_dim, hidden_dims, output_dim)

    # ✅ PyTorch 官方损失函数
    criterion = nn.CrossEntropyLoss()

    # ✅ 提取参数
    params = []
    for layer in model.layers:
        params.append(layer.weights)
        params.append(layer.biases)

    # ✅ PyTorch 官方优化器
    optimizer = optim.Adam(params, lr=0.001)

    epochs = 100
    batch_size = 64
    print("开始训练...")

    for epoch in range(epochs):
        train_loss = 0.0

        for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)

            # 前向
            y_pred = model.forward(batch_x)  # 输出是 softmax
            logits = torch.log(y_pred + 1e-7)  # 转成 logits 适配官方 CE

            # 计算损失
            loss = criterion(logits, batch_y)
            train_loss += loss.item()

            # ✅ PyTorch 自动反向传播（核心！不用手动 backward）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估
        avg_train_loss = train_loss / (len(train_x) / batch_size)
        train_acc = model.accuracy(train_x[:1000], train_y[:1000])
        test_acc = model.accuracy(test_x, test_y)

        print(f"Epoch [{epoch+1}/{epochs}] | 训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

    # 保存权重
    weights_dict = {}
    biases_dict = {}
    for i, layer in enumerate(model.layers):
        weights_dict[f"layer_{i}_weights"] = layer.weights.detach().cpu().numpy()
        biases_dict[f"layer_{i}_biases"] = layer.biases.detach().cpu().numpy()

    np.savez("mlp_weights.npz", **weights_dict)
    np.savez("mlp_biases.npz", **biases_dict)
    print("训练完成，模型权重已保存！")

if __name__ == "__main__":
    train_mlp()