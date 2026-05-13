import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """将图像分割为Patch并编码"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # CIFAR-10: 32/4=8 → 64个patch

        # 卷积实现Patch分割+嵌入
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, 32, 32] → [B, embed_dim, 8, 8] → [B, embed_dim, 64] → [B, 64, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # QKV线性层
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape  # B: batch_size, N: num_patches+1, C: embed_dim

        # 生成QKV: [B, N, 3*C] → [B, N, 3, num_heads, head_dim] → [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算: Q@K^T / √d_k
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出: [B, num_heads, N, head_dim] → [B, N, num_heads*head_dim] → [B, N, C]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, embed_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()  # GELU更适合Transformer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x):
        # 残差连接 + 层归一化
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    视觉Transformer（ViT），专为CIFAR-10优化
    结构: PatchEmbedding → Class Token → Pos Embedding → Transformer Encoder × N → 分类头
    """

    def __init__(self, num_classes=10, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=256, num_heads=8, num_layers=4, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Patch嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class Token（ViT核心，用于分类）
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置嵌入（包含class token）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer编码器
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes)
        )

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(self._init_weights)

        # 自动放到GPU/CPU
        self.to(self.device)

    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        # 形状变换 [B, 3072] → [B, 3, 32, 32]（适配CIFAR-10展平输入）
        x = x.view(-1, 3, 32, 32)

        # Patch嵌入
        x = self.patch_embed(x)  # [B, 64, 256]

        # 添加Class Token
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)  # [B, 1, 256]
        x = torch.cat([class_token, x], dim=1)  # [B, 65, 256]

        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer编码
        x = self.encoder(x)

        # 取Class Token做分类
        x = self.norm(x)
        x = x[:, 0]  # [B, 256]

        # 分类头
        x = self.head(x)
        return x

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out, dim=-1).cpu().numpy()

    def accuracy(self, x, y_true):
        y_pred = self.predict(x)
        return np.mean(y_pred == y_true)

    def __repr__(self):
        return (f"ViT(\n"
                f"  结构: Patch({self.patch_embed.patch_size}×{self.patch_embed.patch_size}) → "
                f"Embed({self.patch_embed.embed_dim}) → Transformer×{len(self.encoder)} → ClassHead\n"
                f"  设备: {self.device}\n"
                f"  Patch数量: {self.patch_embed.num_patches}\n)")