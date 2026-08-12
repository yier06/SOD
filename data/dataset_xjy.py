"""COCO detection dataset and dataloader helpers.

The SODA-D annotations use COCO ``bbox=[x, y, width, height]`` format.
The model consumes absolute ``xyxy`` boxes.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


class COCODetectionDataset(Dataset):
    def __init__(self, image_dir: Path, annotation_file: Path, image_size: int,
                 train: bool = False):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.image_size = image_size
        self.train = train

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"图片目录不存在: {self.image_dir}")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_file}")

        data = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        # SODA-D 的 ``ignore`` 是忽略区域，不应作为可学习类别。
        categories = sorted(
            [item for item in data["categories"] if item["name"].lower() != "ignore"],
            key=lambda item: item["id"],
        )
        self.classes = [item["name"] for item in categories]
        self.class_to_idx = {item["name"]: i for i, item in enumerate(categories)}
        self.category_to_idx = {item["id"]: i for i, item in enumerate(categories)}

        annotations: Dict[int, List[dict]] = {}
        for ann in data.get("annotations", []):
            if ann.get("ignore", 0):
                continue
            annotations.setdefault(ann["image_id"], []).append(ann)

        self.items = []
        missing = 0
        for image in data["images"]:
            path = self.image_dir / image["file_name"]
            if not path.is_file():
                missing += 1
                continue
            self.items.append((image, annotations.get(image["id"], [])))
        if missing:
            print(f"警告: {self.annotation_file.name} 有 {missing} 张图片未找到，已跳过")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_info, anns = self.items[index]
        image = Image.open(self.image_dir / image_info["file_name"]).convert("RGB")
        old_w, old_h = image.size

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(old_w), x + max(0.0, w)), min(float(old_h), y + max(0.0, h))
            if x2 > x1 and y2 > y1 and ann["category_id"] in self.category_to_idx:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.category_to_idx[ann["category_id"]])

        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        sx, sy = self.image_size / old_w, self.image_size / old_h
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if boxes.numel():
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        labels = torch.tensor(labels, dtype=torch.long)

        if self.train and torch.rand(()) < 0.5:
            image = TF.hflip(image)
            if boxes.numel():
                old_x1 = boxes[:, 0].clone()
                boxes[:, 0] = self.image_size - boxes[:, 2]
                boxes[:, 2] = self.image_size - old_x1

        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, {"boxes": boxes, "labels": labels, "image_id": image_info["id"]}


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)


def build_dataloaders(config):
    train_set = COCODetectionDataset(config.image_dir, config.split_annotation("train"), config.image_size, True)
    val_set = COCODetectionDataset(config.image_dir, config.split_annotation("val"), config.image_size, False)
    if train_set.classes != val_set.classes:
        raise ValueError("训练集和验证集类别定义不一致")

    common = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                  pin_memory=torch.cuda.is_available(), collate_fn=detection_collate,
                  persistent_workers=config.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    print(f"训练集: {len(train_set)} 张 | 验证集: {len(val_set)} 张 | 类别: {train_set.classes}")
    return train_loader, val_loader, train_set.classes, train_set.class_to_idx
