# ==============================
# Transformer 奠基
# 理由：基于自注意力机制，完全替代循环/卷积结构，是现代大模型、ViT、扩散模型Transformer 变体的核心基础
# 标题：Attention Is All You Need
# 会议：NeurIPS 2017（原NIPS 2017）
# 单位：Google Brain / Google Research
# 代码：https://github.com/chen6561/Core-NeuralNet/blob/main/examples/run_transformer.py
# 论文：https://arxiv.org/abs/1706.03762
# ==============================

import torch
import torch.nn as nn
import math


# ================================
# 0. 位置编码（论文标准版）
# 功能：给序列加入“位置信息”，因为自注意力本身不知道词的顺序
# 例子：句子 "我爱编程"，自注意力看不到顺序，必须告诉模型 我(1)→爱(2)→编(3)→程(4)
# ================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 位置编码矩阵：max_len=最大句子长度，d_model=词向量维度
        pe = torch.zeros(max_len, d_model)

        # position = 位置序号：0,1,2,3,...,max_len-1
        # 例子：句子第 0 个字、第 1 个字、第 2 个字...
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 分母项，控制正弦余弦的周期
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 偶数维度用 sin
        # 例子：d_model=4 → 维度0、2用sin
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度用 cos
        # 例子：d_model=4 → 维度1、3用cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度，变成 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为缓冲区（不参与训练，但会随模型保存）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：把位置编码直接加到词嵌入上
        :param x: 词向量 [batch_size, seq_len, d_model]
        :return: 带位置信息的词向量
        例子：x = [“我”“爱”“编”“程”] → 加上位置 → 知道这四个字是按顺序来的
        """
        seq_len = x.size(1)  # 句子长度
        return x + self.pe[:, :seq_len]  # 直接相加，位置信息注入


# ================================
# 1. 缩放点积注意力
# 公式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
# 通俗解释：
# Q：我要查什么（查询）
# K：库里有什么（键）
# V：实际内容（值）
# 注意力 = 查出来的内容权重
# ================================
class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        """
        计算注意力
        :param q: 查询 [B, 头数, 句子长度, 向量维度]
        :param k: 键   [B, 头数, 句子长度, 向量维度]
        :param v: 值   [B, 头数, 句子长度, 向量维度]
        :param mask: 掩码，把不该看的位置遮住（比如未来词）
        :return: 注意力输出
        例子：
        Q = “我”
        K = “我”“爱”“编”“程”
        计算 Q 对每个 K 的相似度 → 得到注意力权重
        """
        d_k = q.size(-1)  # 每个头的向量维度，如 64

        # 1. Q × K^T → 相似度分数
        # 例子：句子长度4 → 分数矩阵 [4,4]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. 掩码：把 mask=0 的地方设为 -无穷，softmax 后就变成 0
        # 例子：解码器不能看到未来的词，第3个字不能看第4、5个字
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. softmax 归一化 → 得到注意力权重
        attn = scores.softmax(dim=-1)

        # 4. 权重 × V → 最终注意力输出
        return torch.matmul(attn, v)


# ================================
# 2. 多头注意力
# 功能：把特征分成多组，并行学习不同类型的注意力
# 例子：
# 头1：找主谓关系
# 头2：找修饰关系
# 头3：找上下文关联
# 最后拼起来 = 更全面的理解
# ================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads  # 每个头的维度，如 512/8=64
        self.n_heads = n_heads  # 头数，默认 8

        # Q、K、V 三个线性投影
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # 输出投影层
        self.out = nn.Linear(d_model, d_model)

        # 缩放点积注意力
        self.attn = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        """
        多头注意力前向
        例子：
        输入：[B, seq_len=4, d_model=512]
        分成 8 个头 → 每个头 [B, 4, 64]
        """
        B = q.size(0)  # batch 大小

        # 1. 线性投影 → 拆成多头
        # [B, L, d_model] → [B, 头数, L, d_k]
        q = self.wq(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力
        out = self.attn(q, k, v, mask)

        # 3. 拼接多头：把 8 个头重新拼回 512 维
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)

        # 4. 最后线性层输出
        return self.out(out)


# ================================
# 3. 前馈网络 FFN
# 公式：FFN(x) = max(0, xW1 + b1)W2 + b2
# 结构：线性 → ReLU → Dropout → 线性
# 作用：对每个词单独做特征增强
# ================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 升维：512 → 2048
        self.linear2 = nn.Linear(d_ff, d_model)  # 降维：2048 → 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前馈网络：每个词独立进行非线性变换
        例子：
        x = 词向量 → 升到高维捕捉复杂特征 → 降维回来
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# ================================
# 4. Encoder Layer
# 结构：
# 自注意力 → Add & Norm → FFN → Add & Norm
# 作用：理解输入句子的语义
# ================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)  # 自注意力
        self.ffn = FeedForward(d_model, d_ff, dropout)  # 前馈网络

        # LayerNorm：对每个样本自己做归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        编码器一层
        例子：
        输入：“我爱编程”
        自注意力：每个字关注其他字
        残差：防止梯度消失
        """
        # 1. 自注意力
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))  # 残差 + 归一化

        # 2. 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ================================
# 5. Decoder Layer
# 结构：
# 掩码自注意力 → 交叉注意力 → FFN
# 作用：逐词生成输出句子
# ================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)  # 自注意力（掩码）
        self.cross_attn = MultiHeadAttention(d_model, n_heads)  # 交叉注意力（看encoder）
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None):
        """
        解码器一层
        例子（机器翻译）：
        输入：I love
        输出预测：programming
        掩码自注意力：不能看到未来的词
        交叉注意力：看中文原句“我爱编程”
        """
        # 1. 掩码自注意力：只能看前面的词，不能看后面
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. 交叉注意力：看编码器的输出（原句子）
        cross_out = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + self.dropout2(cross_out))

        # 3. 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# ================================
# 6. 完整 Transformer
# 结构：
# 嵌入 → 位置编码 → Encoder × N → Decoder × N → 线性输出
# 用途：机器翻译、文本生成、对话模型、大语言模型
# ================================
class Transformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=1000):
        super().__init__()
        self.d_model = d_model

        # 词嵌入层：把词编号 → 向量
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # 堆叠 N 层 Encoder
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # 堆叠 N 层 Decoder
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # 最终输出层：预测下一个词
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        完整 Transformer 前向
        :param src: 输入句子编号 [B, src_len]
        :param tgt: 输出句子编号 [B, tgt_len]
        :return: 输出词概率 [B, tgt_len, vocab_size]
        例子（中→英翻译）：
        src = 我爱编程
        tgt = I love
        输出预测：programming
        """
        # 1. 词嵌入 + 缩放（防止值过大）
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)

        # 2. 加入位置信息
        src_emb = self.pos_enc(src_emb)
        tgt_emb = self.pos_enc(tgt_emb)

        # 3. Encoder 前向
        enc_out = src_emb
        for layer in self.encoders:
            enc_out = layer(enc_out, src_mask)

        # 4. Decoder 前向
        dec_out = tgt_emb
        for layer in self.decoders:
            dec_out = layer(dec_out, enc_out, tgt_mask)

        # 5. 输出预测
        output = self.linear(dec_out)
        return output