import numpy as np
from urllib.request import urlretrieve
import tarfile
import os


def load_cifar10(flatten=False):
    """
    加载CIFAR-10数据集（自动下载）
    Args:
        flatten: 是否打平为一维数组（True: MLP用，False: CNN/ViT用）
    Returns:
        (train_x, train_y), (test_x, test_y)
    """
    # CIFAR-10下载地址（官方tar.gz包）
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    data_dir = "data/cifar-10-batches-py"

    # 创建数据目录
    if not os.path.exists("data"):
        os.makedirs("data")

    # 下载数据集（如果不存在）
    if not os.path.exists(os.path.join("data", filename)):
        print("正在下载CIFAR-10数据集...")
        urlretrieve(url, os.path.join("data", filename))

    # 解压数据集（如果未解压）
    if not os.path.exists(data_dir):
        with tarfile.open(os.path.join("data", filename), "r:gz") as tar:
            tar.extractall(path="data")

    # 读取单个批次文件的函数
    def load_batch(filepath):
        import pickle
        with open(filepath, "rb") as f:
            dict = pickle.load(f, encoding="bytes")
        # 图像数据：(10000, 3, 32, 32) → 转换为 (10000, 32, 32, 3)（通道在后，符合常规CV习惯）
        images = dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(dict[b'labels'])
        return images, labels

    # 加载训练集（5个批次，共50000样本）
    train_x = []
    train_y = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        images, labels = load_batch(batch_file)
        train_x.append(images)
        train_y.append(labels)
    train_x = np.concatenate(train_x, axis=0).astype(np.float32) / 255.0  # 归一化到[0,1]
    train_y = np.concatenate(train_y, axis=0)

    # 加载测试集（10000样本）
    test_file = os.path.join(data_dir, "test_batch")
    test_x, test_y = load_batch(test_file)
    test_x = test_x.astype(np.float32) / 255.0
    test_y = test_y.astype(np.int64)

    # 如果需要打平（MLP使用）：32×32×3 → 3072维
    if flatten:
        train_x = train_x.reshape(-1, 32 * 32 * 3)
        test_x = test_x.reshape(-1, 32 * 32 * 3)

    print(f"CIFAR-10加载完成：训练集{train_x.shape}，测试集{test_x.shape}")
    return (train_x, train_y), (test_x, test_y)


def batch_generator(x, y, batch_size=32):
    """生成批次数据（逻辑完全不变）"""
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        batch_indices = indices[start:end]
        yield x[batch_indices], y[batch_indices]