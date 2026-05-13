import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    带防过拟合的卷积神经网络（Dropout + BatchNorm）
    专为 CIFAR-10 优化，解决过拟合问题
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 卷积块：Conv + BatchNorm + ReLU + MaxPool + Dropout
        self.features = nn.Sequential(
            # 块1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # 块2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # 块3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 自动放到 GPU
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        # 形状变换 [B, 3072] -> [B, 3, 32, 32]
        x = x.view(-1, 3, 32, 32)

        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out, dim=-1).cpu().numpy()

    def accuracy(self, x, y_true):
        y_pred = self.predict(x)
        return np.mean(y_pred == y_true)

    def __repr__(self):
        return (f"CNN(\n"
                f"  结构: 3→32→64→128 → 512→10 (带BatchNorm+Dropout)\n"
                f"  设备: {self.device}\n)")