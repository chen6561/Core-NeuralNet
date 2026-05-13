# 导出所有模型（简化导入）
from .mlp import MLP
# 后续添加CNN/Transformer/ViT后，在这里补充：
from .cnn import CNN
from .transformer import Transformer
from .vit import ViT

__all__ = [
    "MLP",
    "CNN",
    "Transformer",
    "ViT"
]