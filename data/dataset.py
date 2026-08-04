import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ..config.dataset.config import Config, set_random_seed

class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, annotations, transforms=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.load_image(index)
        boxes, labels = self.load_annotation(index)

        sample = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

# ============================================================
# 3. 构建数据集和 DataLoader
# ============================================================

def build_dataloaders(config: Config):
    """
    构建训练集、验证集和对应的 DataLoader。
    """

    train_dir = Path(config.train_dir)
    val_dir = Path(config.val_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在：{train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"验证集目录不存在：{val_dir}")

    # 训练集数据增强
    train_transform = transforms.Compose([
        # 随机裁剪，同时缩放到固定大小
        transforms.RandomResizedCrop(
            size=config.image_size,
            scale=(0.8, 1.0)
        ),

        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),

        # 随机调整亮度、对比度和饱和度
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        # PIL Image 转为 Tensor
        # 像素范围从 [0, 255] 转为 [0, 1]
        transforms.ToTensor(),

        # 图像归一化
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集不能使用随机增强
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ImageFolder 会自动根据子文件夹名称生成类别
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

    # 检查训练集和验证集的类别是否一致
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "训练集和验证集类别不一致。\n"
            f"训练集类别：{train_dataset.class_to_idx}\n"
            f"验证集类别：{val_dataset.class_to_idx}"
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print("=" * 60)
    print("数据集加载完成")
    print(f"训练集图片数量：{len(train_dataset)}")
    print(f"验证集图片数量：{len(val_dataset)}")
    print(f"类别数量：{len(train_dataset.classes)}")
    print(f"类别名称：{train_dataset.classes}")
    print(f"类别映射：{train_dataset.class_to_idx}")
    print("=" * 60)

    return (
        train_loader,
        val_loader,
        train_dataset.classes,
        train_dataset.class_to_idx
    )