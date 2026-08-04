from pathlib import Path
import torch.nn as nn
import torch
# ============================================================
# 8. 保存模型
# ============================================================

def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_accuracy: float,
    class_names: list[str],
    class_to_idx: dict[str, int]
):
    """
    保存模型参数和训练信息。
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy,
        "class_names": class_names,
        "class_to_idx": class_to_idx
    }

    torch.save(checkpoint, save_path)