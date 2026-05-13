import torch.optim as optim


# PyTorch原生已实现SGD（带动量）和Adam，直接封装保持接口一致
class SGD:
    """封装PyTorch SGD优化器"""
    def __init__(self, params, learning_rate=0.01, momentum=0.0):
        self.optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)

    def step(self):
        """参数更新"""
        self.optimizer.step()

    def zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()


class Adam:
    """封装PyTorch Adam优化器"""
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.optimizer = optim.Adam(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon
        )

    def step(self):
        """参数更新"""
        self.optimizer.step()

    def zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()