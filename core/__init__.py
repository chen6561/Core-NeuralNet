# 导出核心组件（让用户更便捷地导入）
from .activations import ReLU, Softmax, Sigmoid
from .layers import Dense
from .losses import CrossEntropyLoss, MSELoss
from .optimizers import SGD, Adam

# 包级版本/信息
__version__ = "1.0.0"
__all__ = [
    # 激活函数
    "ReLU", "Softmax", "Sigmoid",
    # 层
    "Dense",
    # 损失函数
    "CrossEntropyLoss", "MSELoss",
    # 优化器
    "SGD", "Adam"
]