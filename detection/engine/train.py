import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...config.dataset.config import Config, set_random_seed
from ...data.dataset_xjy import build_dataloaders
from ..model.model import CustomYOLO11
from ..loss import YOLO11DetectionLoss
from .checkpoint import save_checkpoint


# ============================================================
# 1. 将目标检测标签移动到设备
# ============================================================

def move_targets_to_device(
    targets: Any,
    device: torch.device,
):
    """
    将目标检测标签移动到指定设备。

    支持以下格式：

    格式一：字典列表

    targets = [
        {
            "boxes": Tensor[N, 4],
            "labels": Tensor[N]
        },
        ...
    ]

    格式二：合并后的字典

    targets = {
        "boxes": Tensor[N, 4],
        "labels": Tensor[N],
        "batch_idx": Tensor[N]
    }
    """

    if isinstance(targets, dict):
        return {
            key: value.to(
                device,
                non_blocking=True,
            )
            if isinstance(value, torch.Tensor)
            else value
            for key, value in targets.items()
        }

    if isinstance(targets, (list, tuple)):
        moved_targets = []

        for target in targets:
            if not isinstance(target, dict):
                raise TypeError(
                    "targets 是列表时，"
                    "其中每个元素必须是字典"
                )

            moved_target = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                if isinstance(value, torch.Tensor)
                else value
                for key, value in target.items()
            }

            moved_targets.append(moved_target)

        return moved_targets

    raise TypeError(
        "不支持的 targets 类型："
        f"{type(targets)}"
    )


# ============================================================
# 2. 从损失结果中读取各项损失
# ============================================================

def parse_loss_output(
    loss_output,
    device: torch.device,
):
    """
    统一解析 criterion 的输出。

    推荐 criterion 返回：

    {
        "loss": total_loss,
        "box_loss": box_loss,
        "cls_loss": cls_loss,
        "dfl_loss": dfl_loss
    }
    """

    if isinstance(loss_output, torch.Tensor):
        zero = torch.zeros(
            (),
            device=device,
            dtype=loss_output.dtype,
        )

        return {
            "loss": loss_output,
            "box_loss": zero,
            "cls_loss": zero,
            "dfl_loss": zero,
        }

    if not isinstance(loss_output, dict):
        raise TypeError(
            "criterion 必须返回 Tensor 或字典，"
            f"当前类型为 {type(loss_output)}"
        )

    if "loss" not in loss_output:
        raise KeyError(
            "criterion 返回的字典中必须包含 'loss'"
        )

    total_loss = loss_output["loss"]

    zero = torch.zeros(
        (),
        device=device,
        dtype=total_loss.dtype,
    )

    return {
        "loss": total_loss,
        "box_loss": loss_output.get(
            "box_loss",
            zero,
        ),
        "cls_loss": loss_output.get(
            "cls_loss",
            zero,
        ),
        "dfl_loss": loss_output.get(
            "dfl_loss",
            zero,
        ),
    }


# ============================================================
# 3. 单轮训练
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    max_grad_norm: float = 10.0,
) -> Dict[str, float]:
    """
    完成一轮目标检测训练。

    返回：
        {
            "loss": 平均总损失,
            "box_loss": 平均边界框损失,
            "cls_loss": 平均分类损失,
            "dfl_loss": 平均 DFL 损失
        }
    """

    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    dfl_loss_sum = 0.0

    total_samples = 0

    use_amp = device.type == "cuda"

    for batch_index, batch_data in enumerate(dataloader):

        if len(batch_data) != 2:
            raise ValueError(
                "dataloader 每个 batch 必须返回 "
                "(images, targets)"
            )

        images, targets = batch_data

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = move_targets_to_device(
            targets,
            device,
        )

        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # 前向传播和损失计算
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            # 训练时返回 P3、P4、P5 三个尺度的原始输出
            raw_outputs = model(
                images,
                decode=False,
            )

            loss_output = criterion(
                raw_outputs,
                targets,
            )

            loss_dict = parse_loss_output(
                loss_output,
                device,
            )

            loss = loss_dict["loss"]
            box_loss = loss_dict["box_loss"]
            cls_loss = loss_dict["cls_loss"]
            dfl_loss = loss_dict["dfl_loss"]

        # ----------------------------------------------------
        # 检查损失是否为 NaN 或 Inf
        # ----------------------------------------------------

        if not torch.isfinite(loss):
            raise RuntimeError(
                "检测到非法损失："
                f"loss={loss.detach().item()}, "
                f"batch={batch_index + 1}"
            )

        # ----------------------------------------------------
        # 反向传播
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        # 梯度裁剪
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # 统计损失
        # ----------------------------------------------------

        total_loss_sum += (
            loss.detach().item() * batch_size
        )

        box_loss_sum += (
            box_loss.detach().item() * batch_size
        )

        cls_loss_sum += (
            cls_loss.detach().item() * batch_size
        )

        dfl_loss_sum += (
            dfl_loss.detach().item() * batch_size
        )

        total_samples += batch_size

        # ----------------------------------------------------
        # 打印当前训练状态
        # ----------------------------------------------------

        if (
            (batch_index + 1) % 10 == 0
            or batch_index + 1 == len(dataloader)
        ):
            current_loss = (
                total_loss_sum / total_samples
            )

            current_box_loss = (
                box_loss_sum / total_samples
            )

            current_cls_loss = (
                cls_loss_sum / total_samples
            )

            current_dfl_loss = (
                dfl_loss_sum / total_samples
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"  Batch [{batch_index + 1:4d}/"
                f"{len(dataloader):4d}] "
                f"Loss: {current_loss:.4f} "
                f"Box: {current_box_loss:.4f} "
                f"Cls: {current_cls_loss:.4f} "
                f"DFL: {current_dfl_loss:.4f} "
                f"LR: {current_lr:.8f}"
            )

    if total_samples == 0:
        raise RuntimeError(
            "训练 dataloader 中没有样本"
        )

    return {
        "loss": total_loss_sum / total_samples,
        "box_loss": box_loss_sum / total_samples,
        "cls_loss": cls_loss_sum / total_samples,
        "dfl_loss": dfl_loss_sum / total_samples,
    }


# ============================================================
# 4. 验证模型
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    在验证集上计算目标检测损失。

    当前只统计：
        total loss
        box loss
        cls loss
        dfl loss

    mAP、Precision、Recall 需要额外实现预测解码、NMS 和指标计算。
    """

    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    cls_loss_sum = 0.0
    dfl_loss_sum = 0.0

    total_samples = 0

    use_amp = device.type == "cuda"

    for images, targets in dataloader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = move_targets_to_device(
            targets,
            device,
        )

        batch_size = images.size(0)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            raw_outputs = model(
                images,
                decode=False,
            )

            loss_output = criterion(
                raw_outputs,
                targets,
            )

            loss_dict = parse_loss_output(
                loss_output,
                device,
            )

        loss = loss_dict["loss"]
        box_loss = loss_dict["box_loss"]
        cls_loss = loss_dict["cls_loss"]
        dfl_loss = loss_dict["dfl_loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                "验证阶段检测到非法损失："
                f"{loss.detach().item()}"
            )

        total_loss_sum += (
            loss.detach().item() * batch_size
        )

        box_loss_sum += (
            box_loss.detach().item() * batch_size
        )

        cls_loss_sum += (
            cls_loss.detach().item() * batch_size
        )

        dfl_loss_sum += (
            dfl_loss.detach().item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "验证 dataloader 中没有样本"
        )

    return {
        "loss": total_loss_sum / total_samples,
        "box_loss": box_loss_sum / total_samples,
        "cls_loss": cls_loss_sum / total_samples,
        "dfl_loss": dfl_loss_sum / total_samples,
    }


# ============================================================
# 5. 完整训练流程
# ============================================================

def train(config: Config):
    """
    完整 YOLO11 风格目标检测训练流程。
    """

    set_random_seed(config.seed)

    # --------------------------------------------------------
    # 设备
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"当前训练设备：{device}")

    if device.type == "cuda":
        print(
            "GPU 型号："
            f"{torch.cuda.get_device_name(0)}"
        )

        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # 保存路径
    # --------------------------------------------------------

    save_dir = Path(config.save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        save_dir / config.best_model_name
    )

    # --------------------------------------------------------
    # 数据集
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        class_names,
        class_to_idx,
    ) = build_dataloaders(config)

    num_classes = len(class_names)

    if num_classes <= 0:
        raise RuntimeError(
            "数据集中没有检测类别"
        )

    print(f"检测类别数量：{num_classes}")
    print(f"检测类别名称：{class_names}")

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    model = CustomYOLO11(
        num_classes=num_classes,
        scale="n",
        reg_max=16,
    ).to(device)

    print(model)

    # --------------------------------------------------------
    # YOLO 检测损失
    # --------------------------------------------------------

    criterion = YOLO11DetectionLoss(
        num_classes=num_classes,
        reg_max=16,
        strides=(8, 16, 32),
        box_weight=getattr(
            config,
            "box_loss_weight",
            7.5,
        ),
        cls_weight=getattr(
            config,
            "cls_loss_weight",
            0.5,
        ),
        dfl_weight=getattr(
            config,
            "dfl_loss_weight",
            1.5,
        ),
    ).to(device)

    # --------------------------------------------------------
    # 优化器
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # --------------------------------------------------------
    # 学习率调度器
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6,
        )
    )

    # --------------------------------------------------------
    # 混合精度
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # 验证损失越小越好
    best_val_loss = math.inf
    best_epoch = 0

    # --------------------------------------------------------
    # Epoch 训练
    # --------------------------------------------------------

    for epoch in range(config.num_epochs):

        print()
        print("=" * 80)

        print(
            f"Epoch [{epoch + 1}/"
            f"{config.num_epochs}]"
        )

        print(
            "当前学习率："
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        print("=" * 80)

        # 训练
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            max_grad_norm=getattr(
                config,
                "max_grad_norm",
                10.0,
            ),
        )

        # 验证
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # 每轮结束后更新学习率
        scheduler.step()

        print()
        print(
            "训练集："
            f"Loss={train_metrics['loss']:.4f}, "
            f"Box={train_metrics['box_loss']:.4f}, "
            f"Cls={train_metrics['cls_loss']:.4f}, "
            f"DFL={train_metrics['dfl_loss']:.4f}"
        )

        print(
            "验证集："
            f"Loss={val_metrics['loss']:.4f}, "
            f"Box={val_metrics['box_loss']:.4f}, "
            f"Cls={val_metrics['cls_loss']:.4f}, "
            f"DFL={val_metrics['dfl_loss']:.4f}"
        )

        # ----------------------------------------------------
        # 保存验证损失最低的模型
        # ----------------------------------------------------

        current_val_loss = val_metrics["loss"]

        if current_val_loss < best_val_loss:

            best_val_loss = current_val_loss
            best_epoch = epoch + 1

            save_checkpoint(
                save_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_accuracy=best_val_loss,
                class_names=class_names,
                class_to_idx=class_to_idx,
            )

            print(
                f"保存新的最优模型："
                f"{best_model_path}"
            )

            print(
                "当前最优验证损失："
                f"{best_val_loss:.4f}"
            )

    # --------------------------------------------------------
    # 训练结束
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("训练完成")
    print(f"最优模型路径：{best_model_path}")
    print(f"最优 Epoch：{best_epoch}")
    print(f"最优验证损失：{best_val_loss:.4f}")
    print("=" * 80)
