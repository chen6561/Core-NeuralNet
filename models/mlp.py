import numpy as np
import torch
from core import Dense
from core import ReLU, Softmax


class MLP:
    """
    多层感知机（MLP）分类模型
    支持 PyTorch Tensor 计算与 CUDA 加速
    网络结构：输入层 → 隐藏层(ReLU) → ... → 输出层 → Softmax
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        """
        初始化 MLP 模型

        参数:
            input_dim (int): 输入特征维度，CIFAR-10 展平后为 3072
            hidden_dims (list): 隐藏层维度列表，例如 [128, 64]
            output_dim (int): 输出类别数，CIFAR-10 为 10
            activation (str): 隐藏层激活函数，目前仅支持 relu
        """
        self.layers = []       # 存储所有全连接层
        self.activations = []  # 存储所有激活函数层

        # 自动检测设备：优先使用 CUDA，否则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建隐藏层
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(Dense(prev_dim, dim, weight_init="he"))
            self.activations.append(ReLU())
            prev_dim = dim

        # 构建输出层（无激活，后续接 Softmax）
        self.layers.append(Dense(prev_dim, output_dim))
        self.softmax = Softmax()

    def forward(self, x):
        """
        前向传播
        自动兼容 numpy 数组与 PyTorch Tensor，并迁移至指定设备
        """
        # 将输入数据转换为 Tensor 并迁移到指定设备
        if isinstance(x, np.ndarray):
            out = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            out = x.to(self.device)

        # 隐藏层前向传播（线性变换 + 激活）
        for layer, activation in zip(self.layers[:-1], self.activations):
            out = layer.forward(out)
            out = activation.forward(out)

        # 输出层线性变换
        out = self.layers[-1].forward(out)
        # 最终 Softmax 归一化
        out = self.softmax.forward(out)
        return out

    def backward(self, grad_output):
        """
        反向传播
        从输出层向输入层依次计算梯度
        """
        # Softmax 层反向传播
        grad = self.softmax.backward(grad_output)
        # 输出层反向传播
        grad = self.layers[-1].backward(grad)

        # 隐藏层反向传播（逆序）
        for layer, activation in reversed(list(zip(self.layers[:-1], self.activations))):
            grad = activation.backward(grad)
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        """
        模型预测
        返回预测类别索引（numpy 格式）
        """
        out = self.forward(x)
        return torch.argmax(out, dim=-1).cpu().numpy()

    def accuracy(self, x, y_true):
        """
        计算模型分类准确率
        """
        y_pred = self.predict(x)
        return np.mean(y_pred == y_true)

    def __repr__(self):
        """
        打印模型结构信息
        """
        return (f"MLP(\n"
                f"  输入维度: {self.layers[0].weights.shape[1]}\n"
                f"  网络结构: {' → '.join([str(layer.weights.shape[0]) for layer in self.layers])}\n"
                f"  总层数: {len(self.layers)}\n"
                f")")