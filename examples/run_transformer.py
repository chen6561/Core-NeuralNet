import torch
from models import Transformer

# ================================
# 词典：手动构建小型英文词汇表
# 用于将真实单词 → 数字 token（模型只能处理数字）
# <pad>: 填充符；<sos>: 句子开头；<eos>: 句子结束
# ================================
vocab = [
    "<pad>", "<sos>", "<eos>",
    "i", "love", "you", "deep", "learning",
    "machine", "computer", "vision", "transformer",
    "data", "feature", "network", "train", "test"
]

# 单词 → 数字索引（模型输入必须是数字）
word2idx = {w: i for i, w in enumerate(vocab)}

# 数字索引 → 单词（用于把模型输出变回可读单词）
idx2word = {i: w for i, w in enumerate(vocab)}


# ================================
# 功能：将自然句子（空格分隔）转为模型可接受的 token 序列
# 例子："i love transformer" → [3,4,11]
# ================================
def sentence_to_tokens(sentence):
    # 转小写 + 按空格切分成单词列表
    words = sentence.lower().split()
    # 逐个查词典，把单词变成数字
    return [word2idx[w] for w in words]


# ================================
# 功能：将模型输出的数字 token 序列转回可读句子
# 例子：[3,4,10] → "i love vision"
# ================================
def tokens_to_sentence(tokens):
    words = []
    for t in tokens:
        # 只转换有效词汇，跳过特殊符号
        if t in idx2word and idx2word[t] not in ["<pad>", "<sos>", "<eos>"]:
            words.append(idx2word[t])
    return " ".join(words)


# ================================
# Transformer 模型测试主程序
# 功能：
# 1. 输入真实英文句子
# 2. 转换成数字 token
# 3. 送入 Transformer 做一次完整前向传播
# 4. 输出预测的下一个单词序列
# ================================
if __name__ == '__main__':
    """
    主函数：用真实单词测试 Transformer
    不训练、只测试前向传播，验证结构、维度、流程完全正确
    """

    # --------------------------
    # 模型超参数（完全遵循论文原版配置）
    # --------------------------
    d_model = 512        # 模型向量维度（词向量维度）
    n_heads = 8          # 多头注意力头数
    n_layers = 6         # Encoder 和 Decoder 各 6 层
    d_ff = 2048          # 前馈网络中间升维维度
    vocab_size = len(vocab)  # 真实词典大小

    # --------------------------
    # 真实自然语言输入（可自己随便改）
    # 场景：翻译 / 语言建模
    # --------------------------
    src_sentence = "i love transformer deep learning"   # 源句子（如：英文）
    tgt_sentence = "i love computer vision"             # 目标句子（如：翻译结果）

    # --------------------------
    # 步骤1：句子 → 数字 token 序列
    # --------------------------
    src_tokens = sentence_to_tokens(src_sentence)  # 单词 → 数字
    tgt_tokens = sentence_to_tokens(tgt_sentence)  # 单词 → 数字

    # 转为 PyTorch 张量，并增加 batch 维度 [1, seq_len]
    src = torch.tensor([src_tokens])
    tgt = torch.tensor([tgt_tokens])

    # --------------------------
    # 从张量里获取维度信息（方便打印理解）
    # --------------------------
    batch_size = src.shape[0]     # 批次大小（这里=1）
    src_len = src.shape[1]        # 源句子长度
    tgt_len = tgt.shape[1]        # 目标句子长度

    # --------------------------
    # 步骤2：初始化 Transformer 模型
    # --------------------------
    model = Transformer(d_model, n_heads, n_layers, d_ff, vocab_size)

    # --------------------------
    # 步骤3：模型前向传播（核心推理过程）
    # 输入：源句子 + 目标句子前缀
    # 输出：预测目标句子的下一个词概率分布
    # --------------------------
    output = model(src, tgt)

    # --------------------------
    # 步骤4：输出结果解析（维度 + 真实单词）
    # --------------------------
    print("=" * 70)
    print("📝 源句子（输入）:", src_sentence)
    print("📝 目标句子（输入）:", tgt_sentence)
    print("🔢 src token 序列:", src_tokens)
    print("🔢 tgt token 序列:", tgt_tokens)
    print("-" * 70)
    print("✅ 输入 src 形状 (batch, src_len):", src.shape)
    print("✅ 输入 tgt 形状 (batch, tgt_len):", tgt.shape)
    print("✅ 输出 logits 形状 (batch, tgt_len, vocab_size):", output.shape)

    # --------------------------
    # 步骤5：将模型输出转为可读句子
    # output 是概率分布 → argmax 取最可能的 token
    # --------------------------
    pred_tokens = torch.argmax(output, dim=-1)[0].tolist()  # 取第1条数据
    pred_sentence = tokens_to_sentence(pred_tokens)

    print("\n🎯 模型预测的目标句子:", pred_sentence)
    print("\n✅ Transformer 真实单词输入输出测试成功！")