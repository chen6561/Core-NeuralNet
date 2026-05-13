风格：学术、清晰、结构完整、适合放 CNN / Transformer / ViT 等主流模型从零实现项目。

---

# FundamentalNN
**Fundamental Neural Networks from Scratch • 从零实现主流深度学习模型**

## 简介
本项目使用 **Python + NumPy**（无 PyTorch/TensorFlow 等框架依赖）从零实现当前深度学习领域最主流的神经网络结构，帮助学习者彻底理解模型内部原理、前向传播与反向传播细节，适合入门、教学与原理研究。

## 已实现模型
- CNN（Convolutional Neural Network）卷积神经网络
- Transformer 完整结构（Encoder + Decoder）
- ViT（Vision Transformer）视觉Transformer
- MLP 基础全连接网络
- 基础组件：卷积层、池化、LayerNorm、Multi-Head Attention、Feed Forward、Positional Encoding 等

## 项目特点
- 纯 Python + NumPy 实现，无框架黑盒
- 代码结构清晰、注释详细、易于阅读
- 支持前向传播与反向传播手动推导
- 包含可直接运行的示例与测试代码
- 适合深度学习基础原理学习与教学演示

## 项目结构
```
FundamentalNN/
├── models/              # 模型实现
│   ├── mlp.py           # 基础全连接网络
│   ├── cnn.py           # 卷积神经网络
│   ├── transformer.py   # Transformer 模型
│   └── vit.py           # Vision Transformer
├── layers/              # 基础层组件
│   ├── conv.py
│   ├── attention.py
│   ├── normalization.py
│   └── activation.py
├── utils/               # 工具函数
│   ├── data_loader.py
│   └── optimizer.py
├── examples/            # 使用示例
└── README.md
```

## 快速开始
1. 克隆项目
```bash
git clone https://github.com/xxx/FundamentalNN.git
cd FundamentalNN
```

2. 安装依赖
```bash
pip install numpy
```

3. 运行示例
```bash
python examples/example_cnn.py
python examples/example_transformer.py
python examples/example_vit.py
```

## 适用人群
- 深度学习初学者
- 想彻底理解 CNN / Transformer / ViT 内部原理的开发者
- 机器学习/算法教学人员
- 需要手动推导反向传播的研究与学习场景

## 核心目标
- 去掉框架封装，让神经网络**每一步计算都可见、可理解、可修改**
- 用最简洁的代码还原经典模型核心逻辑
- 帮助学习者建立从数学到代码的直观理解
