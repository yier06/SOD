"""
基础 PyTorch 图像分类训练模板

功能：
1. 使用 ImageFolder 导入数据集
2. 自定义 CNN 网络
3. 训练和验证
4. 自动保存验证集准确率最高的模型
5. 支持 GPU 和混合精度训练
6. 支持单张图片推理
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms














# ============================================================
# 9. 完整训练入口
# ============================================================




# ============================================================
# 10. 单张图片推理
# ============================================================

@torch.no_grad()
def predict_image(
    image_path: str,
    checkpoint_path: str,
    image_size: int = 224
):
    """
    加载训练好的模型，对单张图片进行分类。
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 创建与训练时相同的模型
    model = CustomCNN(
        num_classes=num_classes
    ).to(device)

    # 加载模型参数
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image)

    # [C, H, W] 增加 batch 维度后变成 [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    logits = model(image_tensor)

    # 将 logits 转换为概率
    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = class_names[predicted_index.item()]
    confidence_value = confidence.item()

    print(f"预测类别：{predicted_class}")
    print(f"预测置信度：{confidence_value:.4f}")

    return predicted_class, confidence_value


# ============================================================
# 11. 程序入口
# ============================================================

if __name__ == "__main__":

    config = Config()

    # 开始训练
    train(config)

    # 训练完成后，可以取消下面代码的注释进行推理
    #
    # predict_image(
    #     image_path="./test.jpg",
    #     checkpoint_path="./checkpoints/best_model.pth",
    #     image_size=config.image_size
    # )