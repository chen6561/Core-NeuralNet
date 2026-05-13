import numpy as np
import torch
from core import Dense
from core import ReLU, Softmax


class MLP:
    """多层感知机（MLP）分类模型 —— 兼容 PyTorch + CUDA 版"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        """
        Args:
            input_dim: 输入维度（如MNIST为784）
            hidden_dims: 隐藏层维度列表（如[128, 64]）
            output_dim: 输出维度（分类类别数）
            activation: 隐藏层激活函数（仅支持relu）
        """
        self.layers = []
        self.activations = []

        # 自动使用 CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建输入层到第一个隐藏层
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(Dense(prev_dim, dim, weight_init="he"))
            self.activations.append(ReLU())
            prev_dim = dim

        # 构建输出层（无激活，后续接softmax）
        self.layers.append(Dense(prev_dim, output_dim))
        self.softmax = Softmax()

    def forward(self, x):
        """前向传播（自动兼容 numpy / torch tensor）"""
        # 自动把输入转成 CUDA tensor
        if isinstance(x, np.ndarray):
            out = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            out = x.to(self.device)

        # 原有逻辑完全不变
        for layer, activation in zip(self.layers[:-1], self.activations):
            out = layer.forward(out)
            out = activation.forward(out)
        out = self.layers[-1].forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, grad_output):
        """反向传播（完全不变）"""
        grad = self.softmax.backward(grad_output)
        grad = self.layers[-1].backward(grad)
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            grad = activation.backward(grad)
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        """预测类别（自动兼容 CUDA）"""
        out = self.forward(x)
        return torch.argmax(out, dim=-1).cpu().numpy()

    def accuracy(self, x, y_true):
        """计算准确率（自动兼容 numpy + CUDA）"""
        y_pred = self.predict(x)
        return np.mean(y_pred == y_true)