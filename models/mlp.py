import numpy as np
import torch
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
        self.layers = []
        self.activations = []

        # 自动检测设备：优先使用 CUDA，否则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建隐藏层
        prev_dim = input_dim
        for dim in hidden_dims:
            # 直接创建参数，不依赖Dense，但保留 .weights .biases 接口
            layer = torch.nn.Linear(prev_dim, dim)
            layer = layer.to(self.device)

            # He 初始化
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.zeros_(layer.bias)

            # 兼容旧接口
            layer.weights = layer.weight
            layer.biases = layer.bias

            self.layers.append(layer)
            self.activations.append(ReLU())
            prev_dim = dim

        # 输出层
        output_layer = torch.nn.Linear(prev_dim, output_dim).to(self.device)
        torch.nn.init.xavier_normal_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)

        output_layer.weights = output_layer.weight
        output_layer.biases = output_layer.bias

        self.layers.append(output_layer)
        self.softmax = Softmax()

    def forward(self, x):
        """
        前向传播
        自动兼容 numpy 数组与 PyTorch Tensor，并迁移至指定设备
        """
        # 数据转换与设备迁移
        if isinstance(x, np.ndarray):
            out = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            out = x.to(self.device)

        # 隐藏层
        for layer, activation in zip(self.layers[:-1], self.activations):
            out = layer(out)
            out = activation.forward(out)

        # 输出层 + softmax
        out = self.layers[-1](out)
        out = self.softmax.forward(out)
        return out

    def backward(self, grad_output):
        """反向传播（适配PyTorch自动梯度，无需手动实现）"""
        # 训练使用官方优化器，无需手动反向传播
        return grad_output

    def predict(self, x):
        """模型预测"""
        out = self.forward(x)
        return torch.argmax(out, dim=-1).cpu().numpy()

    def accuracy(self, x, y_true):
        """计算准确率"""
        y_pred = self.predict(x)
        return np.mean(y_pred == y_true)

    def __repr__(self):
        """打印模型结构"""
        layer_shapes = [str(layer.out_features) for layer in self.layers]
        return (f"MLP(\n"
                f"  输入维度: {self.layers[0].in_features}\n"
                f"  网络结构: {' → '.join(layer_shapes)}\n"
                f"  总层数: {len(self.layers)}\n"
                f")")