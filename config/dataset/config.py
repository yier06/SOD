import torch
import numpy as np
import random
# ============================================================
# 配置参数
# ============================================================

class Config():
    # 数据集路径
    train_dir = "./dataset/train"
    val_dir = "./dataset/val"

    # 模型保存路径
    save_dir = "./checkpoints"
    best_model_name = "best_model.pth"

    # 图像尺寸
    image_size = 224

    # 训练参数
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4

    # DataLoader 参数
    # Windows 下出现多进程问题时，可以设为 0
    num_workers = 4

    # 随机种子
    seed = 42

# ============================================================
# 2. 固定随机种子
# ============================================================

def set_random_seed(seed: int) -> None:
    """
    固定随机种子，让每次训练结果尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证卷积计算具有较好的可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
