import torch
import torch.nn as nn

# 全局指定设备（自动用CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dense:
    def __init__(self, input_dim, output_dim, weight_init="xavier"):
        # 🔥 关键修复：权重、偏置 直接创建在 CUDA 上
        self.weights = torch.randn(output_dim, input_dim, device=device)
        self.biases = torch.zeros(output_dim, device=device)

        # 初始化（保持你原来的逻辑）
        if weight_init == "xavier":
            nn.init.xavier_normal_(self.weights)
        elif weight_init == "he":
            nn.init.kaiming_normal_(self.weights, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.normal_(self.weights, mean=0.0, std=0.01)

        # 开启梯度
        self.weights.requires_grad_(True)
        self.biases.requires_grad_(True)

    def forward(self, x):
        # 矩阵运算 x @ W.T + b
        return x @ self.weights.T + self.biases