import torch
from pathlib import Path
from ...data.dataset_xjy import build_dataloaders

from ...config.dataset.config import Config, set_random_seed
from ...data.dataset import build_dataloaders
from .checkpoint import save_checkpoint




# ============================================================
# 6. 单轮训练
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device
):
    """
    完成一轮训练。
    """

    # 切换到训练模式
    # BatchNorm 和 Dropout 会启用训练行为
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(dataloader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # 清空上一轮反向传播留下的梯度
        optimizer.zero_grad(set_to_none=True)

        # 只有 CUDA 环境下才开启混合精度
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda"
        ):
            # 前向传播
            logits = model(images)

            # 计算损失
            loss = criterion(logits, labels)

        # 反向传播
        scaler.scale(loss).backward()

        # 更新模型参数
        scaler.step(optimizer)

        # 更新缩放因子
        scaler.update()

        # 统计损失
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        # logits 中分数最大的位置作为预测类别
        predictions = torch.argmax(logits, dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        # 每隔一定 batch 打印一次
        if (batch_index + 1) % 10 == 0:
            current_loss = total_loss / total_samples
            current_acc = total_correct / total_samples

            print(
                f"  Batch [{batch_index + 1:4d}/"
                f"{len(dataloader):4d}] "
                f"Loss: {current_loss:.4f} "
                f"Accuracy: {current_acc:.4f}"
            )

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


# ============================================================
# 7. 验证模型
# ============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """
    在验证集上评估模型。

    @torch.no_grad() 表示不计算梯度，可以降低显存占用。
    """

    # 切换到评估模式
    # BatchNorm 和 Dropout 使用推理行为
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        predictions = torch.argmax(logits, dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples

    return val_loss, val_accuracy



def train(config: Config):
    """
    完整训练流程。
    """

    set_random_seed(config.seed)

    # 自动选择 GPU 或 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"当前训练设备：{device}")

    if device.type == "cuda":
        print(f"GPU 型号：{torch.cuda.get_device_name(0)}")

    # 创建模型保存目录
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = save_dir / config.best_model_name

    # 加载数据
    (
        train_loader,
        val_loader,
        class_names,
        class_to_idx
    ) = build_dataloaders(config)

    num_classes = len(class_names)

    # 创建模型
    model = CustomCNN(
        num_classes=num_classes
    ).to(device)

    print(model)

    # 交叉熵损失
    # 分类模型最后一层不需要手动添加 Softmax
    criterion = nn.CrossEntropyLoss()

    # AdamW 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )

    # 混合精度梯度缩放器
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    best_accuracy = 0.0

    for epoch in range(config.num_epochs):

        print()
        print("=" * 60)
        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}]"
        )
        print(
            f"当前学习率："
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )
        print("=" * 60)

        # 训练
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )

        # 验证
        val_loss, val_accuracy = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        # 更新学习率
        scheduler.step()

        print(
            f"训练集：Loss={train_loss:.4f}, "
            f"Accuracy={train_accuracy:.4f}"
        )

        print(
            f"验证集：Loss={val_loss:.4f}, "
            f"Accuracy={val_accuracy:.4f}"
        )

        # 验证准确率提高时保存模型
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_checkpoint(
                save_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_accuracy=best_accuracy,
                class_names=class_names,
                class_to_idx=class_to_idx
            )

            print(
                f"保存新的最优模型：{best_model_path}"
            )
            print(
                f"当前最优验证准确率：{best_accuracy:.4f}"
            )

    print()
    print("=" * 60)
    print("训练完成")
    print(f"最优模型路径：{best_model_path}")
    print(f"最优验证准确率：{best_accuracy:.4f}")
    print("=" * 60)