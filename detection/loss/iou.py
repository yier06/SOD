import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    计算 xyxy 格式边界框面积。

    参数：
        boxes: [..., 4]

    返回：
        [...]
    """

    width = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)

    return width * height


def bbox_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    计算逐元素 IoU。

    boxes1 和 boxes2 必须能够广播到相同形状。

    参数：
        boxes1: [..., 4]
        boxes2: [..., 4]

    返回：
        [...]
    """

    intersection_top_left = torch.maximum(
        boxes1[..., :2],
        boxes2[..., :2],
    )

    intersection_bottom_right = torch.minimum(
        boxes1[..., 2:],
        boxes2[..., 2:],
    )

    intersection_wh = (
        intersection_bottom_right
        - intersection_top_left
    ).clamp(min=0)

    intersection = (
        intersection_wh[..., 0]
        * intersection_wh[..., 1]
    )

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    union = area1 + area2 - intersection

    return intersection / (union + eps)


def pairwise_bbox_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    计算两组框之间的两两 IoU。

    参数：
        boxes1: [N, 4]
        boxes2: [M, 4]

    返回：
        [N, M]
    """

    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]

    return bbox_iou(
        boxes1,
        boxes2,
        eps=eps,
    )


def bbox_ciou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    计算 Complete IoU。

    参数：
        boxes1: [..., 4]
        boxes2: [..., 4]

    返回：
        [...]
    """

    iou = bbox_iou(
        boxes1,
        boxes2,
        eps=eps,
    )

    # 预测框宽高
    width1 = (
        boxes1[..., 2]
        - boxes1[..., 0]
    ).clamp(min=eps)

    height1 = (
        boxes1[..., 3]
        - boxes1[..., 1]
    ).clamp(min=eps)

    # 真实框宽高
    width2 = (
        boxes2[..., 2]
        - boxes2[..., 0]
    ).clamp(min=eps)

    height2 = (
        boxes2[..., 3]
        - boxes2[..., 1]
    ).clamp(min=eps)

    # 中心点
    center_x1 = (
        boxes1[..., 0]
        + boxes1[..., 2]
    ) / 2

    center_y1 = (
        boxes1[..., 1]
        + boxes1[..., 3]
    ) / 2

    center_x2 = (
        boxes2[..., 0]
        + boxes2[..., 2]
    ) / 2

    center_y2 = (
        boxes2[..., 1]
        + boxes2[..., 3]
    ) / 2

    center_distance = (
        (center_x1 - center_x2) ** 2
        + (center_y1 - center_y2) ** 2
    )

    # 最小外接矩形
    enclosing_x1 = torch.minimum(
        boxes1[..., 0],
        boxes2[..., 0],
    )

    enclosing_y1 = torch.minimum(
        boxes1[..., 1],
        boxes2[..., 1],
    )

    enclosing_x2 = torch.maximum(
        boxes1[..., 2],
        boxes2[..., 2],
    )

    enclosing_y2 = torch.maximum(
        boxes1[..., 3],
        boxes2[..., 3],
    )

    enclosing_diagonal = (
        (enclosing_x2 - enclosing_x1) ** 2
        + (enclosing_y2 - enclosing_y1) ** 2
        + eps
    )

    # 宽高比惩罚
    v = (
        4
        / (torch.pi ** 2)
        * (
            torch.atan(width2 / height2)
            - torch.atan(width1 / height1)
        ) ** 2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = (
        iou
        - center_distance / enclosing_diagonal
        - alpha * v
    )

    return ciou