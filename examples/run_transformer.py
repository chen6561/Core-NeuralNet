import torch
from models import Transformer

# ================================
# Transformer 模型测试主程序
# 功能：验证模型前向传播、检查维度匹配
# ================================
if __name__ == '__main__':
    """
    主函数：测试 Transformer 模型的前向传播流程
    使用随机生成的序列数据验证网络结构正确性
    """
    # --------------------------
    # 模型超参数（论文原版配置）
    # --------------------------
    d_model = 512        # 模型特征维度
    n_heads = 8          # 多头注意力头数
    n_layers = 6         # Encoder/Decoder 层数
    d_ff = 2048          # 前馈网络中间层维度
    vocab_size = 1000    # 词汇表大小

    # --------------------------
    # 输入数据维度配置
    # --------------------------
    batch_size = 2       # 批次大小
    src_len = 10         # 源序列长度
    tgt_len = 8          # 目标序列长度

    # --------------------------
    # 生成随机测试数据（模拟token序列）
    # --------------------------
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    # --------------------------
    # 初始化 Transformer 模型
    # --------------------------
    model = Transformer(d_model, n_heads, n_layers, d_ff, vocab_size)

    # --------------------------
    # 执行前向传播
    # --------------------------
    output = model(src, tgt)

    # --------------------------
    # 输出维度信息，验证正确性
    # --------------------------
    print("输入 src shape:", src.shape)
    print("输入 tgt shape:", tgt.shape)
    print("输出 logits shape:", output.shape)
    print("\n✅ Transformer 完整运行成功！")