from typing import List, Sequence, Tuple

import torch


def make_anchors(
    feature_shapes: Sequence[Tuple[int, int]],
    strides: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为多个特征层生成 anchor-free 网格点。

    参数：
        feature_shapes:
            [(H3, W3), (H4, W4), (H5, W5)]

        strides:
            [8, 16, 32]

    返回：
        anchor_points:
            [N, 2]
            特征图坐标，不是像素坐标。

        stride_tensor:
            [N, 1]
    """

    all_anchor_points: List[torch.Tensor] = []
    all_stride_tensors: List[torch.Tensor] = []

    for (height, width), stride in zip(
        feature_shapes,
        strides,
    ):
        y = torch.arange(
            height,
            device=device,
            dtype=dtype,
        )

        x = torch.arange(
            width,
            device=device,
            dtype=dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )

        anchor_points = torch.stack(
            [
                grid_x + offset,
                grid_y + offset,
            ],
            dim=-1,
        ).reshape(-1, 2)

        stride_tensor = torch.full(
            size=(height * width, 1),
            fill_value=float(stride),
            device=device,
            dtype=dtype,
        )

        all_anchor_points.append(anchor_points)
        all_stride_tensors.append(stride_tensor)

    return (
        torch.cat(all_anchor_points, dim=0),
        torch.cat(all_stride_tensors, dim=0),
    )


def dist2bbox(
    distances: torch.Tensor,
    anchor_points: torch.Tensor,
) -> torch.Tensor:
    """
    将 ltrb 距离转换为 xyxy 边界框。

    参数：
        distances:
            [B, N, 4] 或 [N, 4]

        anchor_points:
            [N, 2]

    返回：
        [B, N, 4] 或 [N, 4]
    """

    left_top = distances[..., :2]
    right_bottom = distances[..., 2:]

    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom

    return torch.cat(
        [x1y1, x2y2],
        dim=-1,
    )


def bbox2dist(
    anchor_points: torch.Tensor,
    boxes: torch.Tensor,
    reg_max: int,
    eps: float = 0.01,
) -> torch.Tensor:
    """
    将 xyxy 边界框转换为 anchor 点到四条边的距离。

    参数：
        anchor_points:
            [N, 2] 或 [B, N, 2]

        boxes:
            [B, N, 4] 或 [N, 4]

        reg_max:
            DFL 类别数量。

    返回：
        [..., 4]
    """

    left_top = anchor_points - boxes[..., :2]
    right_bottom = boxes[..., 2:] - anchor_points

    distances = torch.cat(
        [left_top, right_bottom],
        dim=-1,
    )

    return distances.clamp(
        min=0,
        max=reg_max - 1 - eps,
    )