风格：学术、清晰、结构完整、适合放 CNN / Transformer / ViT 等主流模型的项目实现。

---

# Core-NeuralNet
**Core-NeuralNet from Scratch • 实现主流深度学习模型**

## 简介
本项目使用 **Python + pytorch**实现当前深度学习领域最主流的神经网络结构，帮助学习者彻底理解模型内部原理、前向传播与反向传播细节，适合入门、教学与原理研究。

## 已实现模型
- CNN（Convolutional Neural Network）卷积神经网络
- Transformer 完整结构（Encoder + Decoder）
- ViT（Vision Transformer）视觉Transformer
- MLP 基础全连接网络
- 基础组件：卷积层、池化、LayerNorm、Multi-Head Attention、Feed Forward、Positional Encoding 等

## 项目特点
- Python + pytroch 实现
- 代码结构清晰、注释详细、易于阅读
- 支持前向传播与反向传播手动推导
- 包含可直接运行的示例与测试代码
- 适合深度学习基础原理学习与教学演示

## 项目结构
```
Core-NeuralNet/
├── README.md                 # 项目介绍、使用方法、实现模型列表
├── requirements.txt          # 依赖：numpy, torch（可选）, matplotlib等
├── core/                     # 核心组件层（所有模型共用的基础模块）
│   ├── __init__.py
│   ├── activations.py        # 激活函数：ReLU, GELU, Softmax等
│   ├── layers.py             # 基础层：Dense, Conv2D, BatchNorm, LayerNorm等
│   ├── attention.py          # 注意力机制：Self-Attention, Multi-Head Attention
│   ├── losses.py              # 损失函数：CrossEntropy, MSE等
│   └── optimizers.py         # 优化器：SGD, Adam等
├── models/                    # 模型实现（你的核心文件都在这里）
│   ├── __init__.py
│   ├── mlp.py                 # 基础全连接网络
│   ├── cnn.py                 # 卷积神经网络
│   ├── transformer.py         # Transformer完整实现（Encoder/Decoder）
│   └── vit.py                 # Vision Transformer实现
├── utils/                     # 工具函数
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载与预处理
│   └── visualizer.py          # 训练过程可视化、注意力热力图等
├── examples/                  # 可直接运行的示例脚本
│   ├── run_mlp.py
│   ├── run_cnn.py
│   ├── run_transformer.py
│   └── run_vit.py
└── tests/                     # 单元测试（验证各模块正确性）
    ├── test_layers.py
    ├── test_attention.py
    └── test_models.py
```

## 快速开始
1. 克隆项目
```bash
git clone https://github.com/chen6561/Core-NeuralNet
cd Core-NeuralNet
```

2. 安装依赖
```bash
pip install numpy
```

3. 运行示例
```bash
python examples/run_mlp.py
python examples/run_cnn.py
python examples/run_transformer.py
python examples/run_vit.py
```

## 适用人群
- 深度学习初学者
- 想彻底理解 CNN / Transformer / ViT 内部原理的开发者
- 机器学习/算法教学人员

## 核心目标
- 用最简洁的代码还原经典模型核心逻辑
- 帮助学习者建立从数学到代码的直观理解
