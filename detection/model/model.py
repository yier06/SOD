import math
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基础模块
# ============================================================

def autopad(
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int], None] = None,
    dilation: int = 1,
):
    """
    自动计算 same padding。
    """
    if padding is not None:
        return padding

    if isinstance(kernel_size, int):
        effective_kernel = dilation * (kernel_size - 1) + 1
        return effective_kernel // 2

    return tuple(
        (dilation * (k - 1) + 1) // 2
        for k in kernel_size
    )


class ConvBNAct(nn.Module):
    """
    Conv2d + BatchNorm2d + SiLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseConv(nn.Module):
    """
    深度可分离卷积：
        Depthwise Conv
        Pointwise Conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )

        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ============================================================
# 2. Bottleneck 与 C3k2
# ============================================================

class Bottleneck(nn.Module):
    """
    残差瓶颈模块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        kernel_size: int = 3,
    ):
        super().__init__()

        hidden_channels = max(1, int(out_channels * expansion))

        self.conv1 = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.conv2 = ConvBNAct(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
        )

        self.use_shortcut = (
            shortcut and in_channels == out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.conv1(x))

        if self.use_shortcut:
            output = output + x

        return output


class C3k(nn.Module):
    """
    C3 风格模块。

    输入被拆成两条路径：
    1. 一条经过多个 Bottleneck
    2. 一条直接进行特征映射
    最后拼接融合
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        kernel_size: int = 3,
    ):
        super().__init__()

        hidden_channels = max(1, int(out_channels * expansion))

        self.branch1 = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.branch2 = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.blocks = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    expansion=1.0,
                    kernel_size=kernel_size,
                )
                for _ in range(num_blocks)
            ]
        )

        self.fusion = ConvBNAct(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.blocks(self.branch1(x))
        branch2 = self.branch2(x)

        output = torch.cat([branch1, branch2], dim=1)
        output = self.fusion(output)

        return output


class C3k2(nn.Module):
    """
    YOLO11 风格 C3k2 模块。

    先将输入映射为两个分支，随后不断追加中间特征，
    最终将所有分支拼接并融合。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = False,
        expansion: float = 0.5,
        use_c3k: bool = False,
    ):
        super().__init__()

        self.hidden_channels = max(
            1,
            int(out_channels * expansion)
        )

        self.input_conv = ConvBNAct(
            in_channels,
            self.hidden_channels * 2,
            kernel_size=1,
        )

        block_class = C3k if use_c3k else Bottleneck

        blocks = []

        for _ in range(num_blocks):
            if use_c3k:
                block = block_class(
                    self.hidden_channels,
                    self.hidden_channels,
                    num_blocks=2,
                    shortcut=shortcut,
                    expansion=1.0,
                    kernel_size=3,
                )
            else:
                block = block_class(
                    self.hidden_channels,
                    self.hidden_channels,
                    shortcut=shortcut,
                    expansion=1.0,
                    kernel_size=3,
                )

            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        fusion_channels = (
            2 + num_blocks
        ) * self.hidden_channels

        self.output_conv = ConvBNAct(
            fusion_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        features = list(torch.chunk(x, chunks=2, dim=1))

        for block in self.blocks:
            features.append(block(features[-1]))

        output = torch.cat(features, dim=1)
        output = self.output_conv(output)

        return output


# ============================================================
# 3. SPPF 空间金字塔池化
# ============================================================

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast。

    通过连续三次 MaxPool 获得不同感受野。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ):
        super().__init__()

        hidden_channels = in_channels // 2

        self.input_conv = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.output_conv = ConvBNAct(
            hidden_channels * 4,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        output = torch.cat(
            [x, pool1, pool2, pool3],
            dim=1,
        )

        return self.output_conv(output)


# ============================================================
# 4. PSA 注意力模块
# ============================================================

class PSAAttention(nn.Module):
    """
    简化的 Position-Sensitive Attention。

    使用多头自注意力处理二维特征图。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        attention_ratio: float = 0.5,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} 必须能被 "
                f"num_heads={num_heads} 整除"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        key_dim = max(
            1,
            int(self.head_dim * attention_ratio)
        )

        self.key_dim = key_dim
        self.scale = key_dim ** -0.5

        query_key_channels = num_heads * key_dim
        value_channels = channels

        self.qkv = ConvBNAct(
            channels,
            query_key_channels * 2 + value_channels,
            kernel_size=1,
            activation=False,
        )

        self.position_encoding = DepthwiseConv(
            channels,
            channels,
            kernel_size=3,
        )

        self.projection = ConvBNAct(
            channels,
            channels,
            kernel_size=1,
            activation=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        num_positions = height * width

        qkv = self.qkv(x)

        q_channels = self.num_heads * self.key_dim
        k_channels = self.num_heads * self.key_dim

        query, key, value = torch.split(
            qkv,
            [
                q_channels,
                k_channels,
                self.channels,
            ],
            dim=1,
        )

        query = query.reshape(
            batch_size,
            self.num_heads,
            self.key_dim,
            num_positions,
        )

        key = key.reshape(
            batch_size,
            self.num_heads,
            self.key_dim,
            num_positions,
        )

        value = value.reshape(
            batch_size,
            self.num_heads,
            self.head_dim,
            num_positions,
        )

        attention = torch.matmul(
            query.transpose(-2, -1),
            key,
        ) * self.scale

        attention = attention.softmax(dim=-1)

        output = torch.matmul(
            value,
            attention.transpose(-2, -1),
        )

        output = output.reshape(
            batch_size,
            self.channels,
            height,
            width,
        )

        output = (
            output
            + self.position_encoding(value.reshape(
                batch_size,
                self.channels,
                height,
                width,
            ))
        )

        return self.projection(output)


class PSABlock(nn.Module):
    """
    PSA 注意力 + 前馈网络。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        shortcut: bool = True,
    ):
        super().__init__()

        self.attention = PSAAttention(
            channels,
            num_heads=num_heads,
        )

        self.feed_forward = nn.Sequential(
            ConvBNAct(
                channels,
                channels * 2,
                kernel_size=1,
            ),
            ConvBNAct(
                channels * 2,
                channels,
                kernel_size=1,
                activation=False,
            ),
        )

        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut:
            x = x + self.attention(x)
            x = x + self.feed_forward(x)
            return x

        x = self.attention(x)
        x = self.feed_forward(x)

        return x


class C2PSA(nn.Module):
    """
    YOLO11 风格 C2PSA 模块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()

        if in_channels != out_channels:
            raise ValueError(
                "C2PSA 要求 in_channels == out_channels"
            )

        hidden_channels = max(
            1,
            int(out_channels * expansion)
        )

        possible_heads = [8, 4, 2, 1]
        num_heads = next(
            head
            for head in possible_heads
            if hidden_channels % head == 0
        )

        self.input_conv = ConvBNAct(
            in_channels,
            hidden_channels * 2,
            kernel_size=1,
        )

        self.blocks = nn.Sequential(
            *[
                PSABlock(
                    hidden_channels,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_conv = ConvBNAct(
            hidden_channels * 2,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1, branch2 = torch.chunk(
            self.input_conv(x),
            chunks=2,
            dim=1,
        )

        branch2 = self.blocks(branch2)

        output = torch.cat(
            [branch1, branch2],
            dim=1,
        )

        return self.output_conv(output)


# ============================================================
# 5. YOLO11 Backbone
# ============================================================

class YOLO11Backbone(nn.Module):
    """
    YOLO11 风格主干网络。

    输出：
        P3: 1/8 尺度
        P4: 1/16 尺度
        P5: 1/32 尺度
    """

    def __init__(
        self,
        channels: Sequence[int],
        depths: Sequence[int],
    ):
        super().__init__()

        if len(channels) != 5:
            raise ValueError("channels 必须包含 5 个通道数")

        if len(depths) != 4:
            raise ValueError("depths 必须包含 4 个重复次数")

        c1, c2, c3, c4, c5 = channels
        d2, d3, d4, d5 = depths

        # P1 / 2
        self.stage1 = ConvBNAct(
            3,
            c1,
            kernel_size=3,
            stride=2,
        )

        # P2 / 4
        self.stage2 = nn.Sequential(
            ConvBNAct(
                c1,
                c2,
                kernel_size=3,
                stride=2,
            ),
            C3k2(
                c2,
                c3,
                num_blocks=d2,
                shortcut=False,
                expansion=0.25,
            ),
        )

        # P3 / 8
        self.stage3 = nn.Sequential(
            ConvBNAct(
                c3,
                c3,
                kernel_size=3,
                stride=2,
            ),
            C3k2(
                c3,
                c4,
                num_blocks=d3,
                shortcut=False,
                expansion=0.25,
            ),
        )

        # P4 / 16
        self.stage4 = nn.Sequential(
            ConvBNAct(
                c4,
                c4,
                kernel_size=3,
                stride=2,
            ),
            C3k2(
                c4,
                c4,
                num_blocks=d4,
                shortcut=True,
                expansion=0.5,
                use_c3k=True,
            ),
        )

        # P5 / 32
        self.stage5 = nn.Sequential(
            ConvBNAct(
                c4,
                c5,
                kernel_size=3,
                stride=2,
            ),
            C3k2(
                c5,
                c5,
                num_blocks=d5,
                shortcut=True,
                expansion=0.5,
                use_c3k=True,
            ),
            SPPF(
                c5,
                c5,
                kernel_size=5,
            ),
            C2PSA(
                c5,
                c5,
                num_blocks=max(1, d5),
                expansion=0.5,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.stage1(x)
        x = self.stage2(x)

        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)

        return p3, p4, p5


# ============================================================
# 6. PAN-FPN Neck
# ============================================================

class YOLO11Neck(nn.Module):
    """
    自顶向下 FPN + 自底向上 PAN。

    输入：
        P3, P4, P5

    输出：
        N3, N4, N5
    """

    def __init__(
        self,
        p3_channels: int,
        p4_channels: int,
        p5_channels: int,
        num_blocks: int = 2,
    ):
        super().__init__()

        # ----------------------------
        # Top-down FPN
        # ----------------------------

        self.p5_reduce = ConvBNAct(
            p5_channels,
            p4_channels,
            kernel_size=1,
        )

        self.p4_fusion = C3k2(
            p4_channels + p4_channels,
            p4_channels,
            num_blocks=num_blocks,
            shortcut=False,
            expansion=0.5,
        )

        self.p4_reduce = ConvBNAct(
            p4_channels,
            p3_channels,
            kernel_size=1,
        )

        self.p3_fusion = C3k2(
            p3_channels + p3_channels,
            p3_channels,
            num_blocks=num_blocks,
            shortcut=False,
            expansion=0.5,
        )

        # ----------------------------
        # Bottom-up PAN
        # ----------------------------

        self.p3_downsample = ConvBNAct(
            p3_channels,
            p3_channels,
            kernel_size=3,
            stride=2,
        )

        self.n4_fusion = C3k2(
            p3_channels + p4_channels,
            p4_channels,
            num_blocks=num_blocks,
            shortcut=False,
            expansion=0.5,
        )

        self.n4_downsample = ConvBNAct(
            p4_channels,
            p4_channels,
            kernel_size=3,
            stride=2,
        )

        self.n5_fusion = C3k2(
            p4_channels + p5_channels,
            p5_channels,
            num_blocks=num_blocks,
            shortcut=True,
            expansion=0.5,
            use_c3k=True,
        )

    def forward(
        self,
        features: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p3, p4, p5 = features

        # ----------------------------
        # Top-down
        # ----------------------------

        p5_reduced = self.p5_reduce(p5)

        p5_up = F.interpolate(
            p5_reduced,
            size=p4.shape[-2:],
            mode="nearest",
        )

        fpn_p4 = self.p4_fusion(
            torch.cat([p5_up, p4], dim=1)
        )

        p4_reduced = self.p4_reduce(fpn_p4)

        p4_up = F.interpolate(
            p4_reduced,
            size=p3.shape[-2:],
            mode="nearest",
        )

        n3 = self.p3_fusion(
            torch.cat([p4_up, p3], dim=1)
        )

        # ----------------------------
        # Bottom-up
        # ----------------------------

        n3_down = self.p3_downsample(n3)

        n4 = self.n4_fusion(
            torch.cat([n3_down, fpn_p4], dim=1)
        )

        n4_down = self.n4_downsample(n4)

        n5 = self.n5_fusion(
            torch.cat([n4_down, p5], dim=1)
        )

        return n3, n4, n5


# ============================================================
# 7. DFL：Distribution Focal Loss 表示解码
# ============================================================

class DFL(nn.Module):
    """
    将离散分布转换为连续距离。

    输入：
        [B, 4 * reg_max, H, W]

    输出：
        [B, 4, H, W]
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()

        self.reg_max = reg_max

        projection = torch.arange(
            reg_max,
            dtype=torch.float32,
        )

        self.register_buffer(
            "projection",
            projection,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        x = x.reshape(
            batch_size,
            4,
            self.reg_max,
            height,
            width,
        )

        x = x.softmax(dim=2)

        projection = self.projection.to(
            dtype=x.dtype,
            device=x.device,
        )

        x = (
            x
            * projection.view(1, 1, -1, 1, 1)
        ).sum(dim=2)

        return x


# ============================================================
# 8. 解耦检测头
# ============================================================

class DecoupledDetectionHead(nn.Module):
    """
    单尺度解耦检测头。

    回归分支输出：
        4 * reg_max

    分类分支输出：
        num_classes
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        reg_max: int = 16,
    ):
        super().__init__()

        regression_channels = max(
            64,
            in_channels // 4,
            reg_max * 4,
        )

        classification_channels = max(
            in_channels,
            min(num_classes * 2, 128),
        )

        self.regression_branch = nn.Sequential(
            ConvBNAct(
                in_channels,
                regression_channels,
                kernel_size=3,
            ),
            ConvBNAct(
                regression_channels,
                regression_channels,
                kernel_size=3,
            ),
            nn.Conv2d(
                regression_channels,
                4 * reg_max,
                kernel_size=1,
            ),
        )

        self.classification_branch = nn.Sequential(
            DepthwiseConv(
                in_channels,
                classification_channels,
                kernel_size=3,
            ),
            DepthwiseConv(
                classification_channels,
                classification_channels,
                kernel_size=3,
            ),
            nn.Conv2d(
                classification_channels,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        regression = self.regression_branch(x)
        classification = self.classification_branch(x)

        return regression, classification


class YOLO11Detect(nn.Module):
    """
    三尺度 YOLO 检测头。

    训练模式默认返回每个尺度的原始输出；
    推理时可以调用 decode() 转换成实际坐标。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Sequence[int],
        strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
    ):
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError(
                "检测头需要 3 个尺度的输入通道"
            )

        if len(strides) != 3:
            raise ValueError(
                "strides 必须包含 3 个步长"
            )

        self.num_classes = num_classes
        self.reg_max = reg_max

        self.register_buffer(
            "strides",
            torch.tensor(
                strides,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.heads = nn.ModuleList(
            [
                DecoupledDetectionHead(
                    channels,
                    num_classes=num_classes,
                    reg_max=reg_max,
                )
                for channels in in_channels
            ]
        )

        self.dfl = DFL(reg_max)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:

        outputs = []

        for feature, head in zip(features, self.heads):
            regression, classification = head(feature)

            outputs.append(
                {
                    "regression": regression,
                    "classification": classification,
                }
            )

        return outputs

    @staticmethod
    def make_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        创建网格中心坐标。

        输出形状：
            [1, H * W, 2]
        """

        y, x = torch.meshgrid(
            torch.arange(
                height,
                device=device,
                dtype=dtype,
            ),
            torch.arange(
                width,
                device=device,
                dtype=dtype,
            ),
            indexing="ij",
        )

        grid = torch.stack(
            [x, y],
            dim=-1,
        )

        grid = grid.reshape(1, -1, 2)

        # 每个网格的中心位置
        return grid + 0.5

    def decode(
        self,
        raw_outputs: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        将原始检测输出解码为：

        [
            x1,
            y1,
            x2,
            y2,
            class_score_1,
            ...,
            class_score_n
        ]

        返回：
            [B, total_points, 4 + num_classes]

        注意：
            这里不执行 NMS。
        """

        decoded_outputs = []

        for level, output in enumerate(raw_outputs):
            regression = output["regression"]
            classification = output["classification"]

            batch_size, _, height, width = regression.shape

            distances = self.dfl(regression)

            distances = distances.permute(
                0, 2, 3, 1
            ).reshape(
                batch_size,
                height * width,
                4,
            )

            class_scores = classification.sigmoid()

            class_scores = class_scores.permute(
                0, 2, 3, 1
            ).reshape(
                batch_size,
                height * width,
                self.num_classes,
            )

            grid = self.make_grid(
                height,
                width,
                regression.device,
                regression.dtype,
            )

            stride = self.strides[level].to(
                device=regression.device,
                dtype=regression.dtype,
            )

            left = distances[..., 0:1]
            top = distances[..., 1:2]
            right = distances[..., 2:3]
            bottom = distances[..., 3:4]

            center_x = grid[..., 0:1]
            center_y = grid[..., 1:2]

            x1 = (center_x - left) * stride
            y1 = (center_y - top) * stride
            x2 = (center_x + right) * stride
            y2 = (center_y + bottom) * stride

            boxes = torch.cat(
                [x1, y1, x2, y2],
                dim=-1,
            )

            prediction = torch.cat(
                [boxes, class_scores],
                dim=-1,
            )

            decoded_outputs.append(prediction)

        return torch.cat(decoded_outputs, dim=1)


# ============================================================
# 9. 完整 YOLO11 检测模型
# ============================================================

class CustomYOLO11(nn.Module):
    """
    从零搭建的 YOLO11 风格目标检测模型。

    参数：
        num_classes:
            检测类别数。

        scale:
            模型规模：
            "n", "s", "m", "l", "x"

        reg_max:
            DFL 离散区间数量，默认 16。

    输入：
        [B, 3, H, W]

    训练输出：
        [
            {
                "regression": [B, 4 * reg_max, H/8, W/8],
                "classification": [B, C, H/8, W/8]
            },
            ...
        ]

    解码后输出：
        [B, N, 4 + num_classes]
    """

    MODEL_SCALES = {
        # depth, width, max_channels
        "n": (0.50, 0.25, 1024),
        "s": (0.50, 0.50, 1024),
        "m": (0.50, 1.00, 512),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.50, 512),
    }

    def __init__(
        self,
        num_classes: int,
        scale: str = "n",
        reg_max: int = 16,
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes 必须大于 0"
            )

        if scale not in self.MODEL_SCALES:
            raise ValueError(
                f"不支持 scale={scale}，"
                f"可选值为 {list(self.MODEL_SCALES.keys())}"
            )

        self.num_classes = num_classes
        self.scale = scale
        self.reg_max = reg_max

        depth_multiple, width_multiple, max_channels = (
            self.MODEL_SCALES[scale]
        )

        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [2, 2, 2, 2]

        channels = [
            self.make_divisible(
                min(channel, max_channels)
                * width_multiple,
                divisor=8,
            )
            for channel in base_channels
        ]

        depths = [
            max(
                1,
                int(round(depth * depth_multiple)),
            )
            for depth in base_depths
        ]

        c1, c2, c3, c4, c5 = channels

        self.backbone = YOLO11Backbone(
            channels=channels,
            depths=depths,
        )

        self.neck = YOLO11Neck(
            p3_channels=c4,
            p4_channels=c4,
            p5_channels=c5,
            num_blocks=max(1, depths[1]),
        )

        self.detect = YOLO11Detect(
            num_classes=num_classes,
            in_channels=(c4, c4, c5),
            strides=(8, 16, 32),
            reg_max=reg_max,
        )

        self._initialize_weights()
        self._initialize_detection_biases()

    @staticmethod
    def make_divisible(
        value: float,
        divisor: int = 8,
    ) -> int:
        """
        将通道数调整为 divisor 的整数倍。
        """
        return max(
            divisor,
            int(value + divisor / 2) // divisor * divisor,
        )

    def _initialize_weights(self) -> None:
        """
        初始化模型参数。
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _initialize_detection_biases(self) -> None:
        """
        初始化检测头偏置。

        分类分支使用较低的初始目标概率，
        避免训练初期产生大量高置信度预测。
        """

        prior_probability = 0.01
        classification_bias = -math.log(
            (1 - prior_probability)
            / prior_probability
        )

        for head in self.detect.heads:
            regression_output = head.regression_branch[-1]
            classification_output = head.classification_branch[-1]

            nn.init.constant_(
                regression_output.bias,
                1.0,
            )

            nn.init.constant_(
                classification_output.bias,
                classification_bias,
            )

    def forward(
        self,
        x: torch.Tensor,
        decode: bool = False,
    ):
        """
        参数：
            x:
                输入图像，形状 [B, 3, H, W]

            decode:
                False：返回训练所需的原始输出
                True：返回解码后的检测框和类别分数
        """

        if x.ndim != 4:
            raise ValueError(
                "输入必须是四维张量 [B, C, H, W]"
            )

        if x.shape[1] != 3:
            raise ValueError(
                f"模型要求 3 通道输入，当前为 {x.shape[1]} 通道"
            )

        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        raw_outputs = self.detect(neck_features)

        if decode:
            return self.detect.decode(raw_outputs)

        return raw_outputs


# ============================================================
# 10. 简单测试
# ============================================================

if __name__ == "__main__":
    num_classes = 20

    model = CustomYOLO11(
        num_classes=num_classes,
        scale="n",
        reg_max=16,
    )

    model.eval()

    images = torch.randn(
        2,
        3,
        640,
        640,
    )

    with torch.no_grad():
        raw_outputs = model(
            images,
            decode=False,
        )

        predictions = model(
            images,
            decode=True,
        )

    print("原始多尺度输出：")

    for index, output in enumerate(raw_outputs):
        print(
            f"尺度 {index}:",
            "regression =",
            tuple(output["regression"].shape),
            "classification =",
            tuple(output["classification"].shape),
        )

    print(
        "解码后的预测形状：",
        tuple(predictions.shape),
    )