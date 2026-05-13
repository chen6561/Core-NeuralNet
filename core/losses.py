import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """交叉熵损失（适配原实现的输入形式）"""
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: 模型输出 (batch_size, num_classes)（未过softmax）
            y_true: 真实标签 (batch_size,) 或 one-hot编码 (batch_size, num_classes)
        """
        # 处理one-hot编码的真实标签（转为类别索引）
        if len(y_true.shape) == 2 and y_true.shape[1] > 1:
            y_true = torch.argmax(y_true, dim=1)
        # PyTorch的CrossEntropyLoss已包含softmax，直接传入原始logits
        return super().forward(y_pred, y_true)


class MSELoss(nn.MSELoss):
    """均方误差损失（复用PyTorch原生实现）"""
    def forward(self, y_pred, y_true):
        return super().forward(y_pred, y_true)