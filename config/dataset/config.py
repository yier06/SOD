"""项目训练配置与随机种子工具。"""

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch


@dataclass
class Config:
    # COCO 数据集目录
    dataset_root: Path = Path("/home/SODA-D")
    image_dir: Path = Path("/home/SODA-D/Images/Images")
    annotation_dir: Path = Path("/home/SODA-D/Annotations/Annotations")

    # 输出与模型
    save_dir: Path = Path("runs/soda_yolo11")
    best_model_name: str = "best.pt"
    last_model_name: str = "last.pt"
    model_scale: str = "n"

    # 输入与训练
    image_size: int = 640
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    num_workers: int = 4
    amp: bool = True
    max_grad_norm: float = 10.0
    seed: int = 42

    # 损失权重
    box_loss_weight: float = 7.5
    cls_loss_weight: float = 0.5
    dfl_loss_weight: float = 1.5

    def split_annotation(self, split: str) -> Path:
        return self.annotation_dir / f"{split}.json"

    def validate(self) -> None:
        if self.image_size % 32 != 0:
            raise ValueError("image_size 必须是 32 的整数倍")
        if self.batch_size < 1 or self.num_epochs < 1:
            raise ValueError("batch_size 和 num_epochs 必须大于 0")
        if self.model_scale not in {"n", "s", "m", "l", "x"}:
            raise ValueError("model_scale 必须是 n/s/m/l/x")


def set_random_seed(seed: int) -> None:
    """固定随机性；训练时仍可通过 cudnn benchmark 换取速度。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
