import numpy as np
from models.mlp import MLP

def test_mlp_forward():
    """测试MLP前向传播维度正确性"""
    # 初始化模型：输入10维，隐藏层[5,3]，输出2类
    model = MLP(input_dim=10, hidden_dims=[5,3], output_dim=2)
    # 生成测试数据（批量大小4）
    x = np.random.randn(4, 10)
    # 前向传播
    out = model.forward(x)
    # 验证输出维度
    assert out.shape == (4, 2), f"输出维度错误，期望(4,2)，实际{out.shape}"
    # 验证softmax概率和为1
    assert np.allclose(np.sum(out, axis=-1), np.ones(4)), "Softmax概率和不为1"
    print("test_mlp_forward 测试通过！")

def test_mlp_predict():
    """测试MLP预测功能"""
    model = MLP(input_dim=5, hidden_dims=[4], output_dim=3)
    x = np.random.randn(2, 5)
    pred = model.predict(x)
    # 验证预测结果是整数，且在0-2范围内
    assert pred.dtype == np.int64, "预测结果类型错误"
    assert np.all(pred >= 0) and np.all(pred < 3), "预测类别超出范围"
    print("test_mlp_predict 测试通过！")

if __name__ == "__main__":
    test_mlp_forward()
    test_mlp_predict()