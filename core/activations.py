import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.ReLU):
    """ReLU激活函数（复用PyTorch原生实现）"""
    def forward(self, x):
        return super().forward(x)


class Softmax(nn.Softmax):
    """Softmax激活函数（适配维度）"""
    def __init__(self):
        super().__init__(dim=-1)  # 沿最后一维计算softmax，与原实现一致

    def forward(self, x):
        return super().forward(x)


class Sigmoid(nn.Sigmoid):
    """Sigmoid激活函数（复用PyTorch原生实现）"""
    def forward(self, x):
        return super().forward(x)