import torch
import torch.nn as nn

# ============================================================
# 4. 自定义基础卷积模块
# ============================================================

class ConvBlock(nn.Module):
    """
    基础卷积模块：

    输入
      ↓
    Conv2d
      ↓
    BatchNorm2d
      ↓
    ReLU
      ↓
    MaxPool2d
      ↓
    输出
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pool: bool = True
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if use_pool:
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# 5. 个性化搭建自己的模型
# ============================================================

class CustomCNN(nn.Module):
    """
    一个自定义 CNN 分类模型。

    输入图像：
        [batch_size, 3, 224, 224]

    输出：
        [batch_size, num_classes]
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # 特征提取部分
        self.features = nn.Sequential(
            # [B, 3, 224, 224]
            ConvBlock(3, 32),

            # [B, 32, 112, 112]
            ConvBlock(32, 64),

            # [B, 64, 56, 56]
            ConvBlock(64, 128),

            # [B, 128, 28, 28]
            ConvBlock(128, 256),

            # [B, 256, 14, 14]
            ConvBlock(256, 512),

            # [B, 512, 7, 7]
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 分类部分
        self.classifier = nn.Sequential(
            # [B, 512]
            nn.Flatten(),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # 输出每个类别的原始分数 logits
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        初始化网络参数。
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        output = self.classifier(features)
        return output

