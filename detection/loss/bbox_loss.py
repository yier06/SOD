from typing import Tuple

import torch
import torch.nn as nn

from .anchor import bbox2dist
from .dfl import DistributionFocalLoss
from .iou import bbox_ciou


class BboxLoss(nn.Module):
    """
    边界框损失：

    1. CIoU Loss
    2. Distribution Focal Loss
    """

    def __init__(
        self,
        reg_max: int = 16,
    ):
        super().__init__()

        self.reg_max = reg_max
        self.dfl_loss = DistributionFocalLoss(
            reg_max=reg_max,
        )

    def forward(
        self,
        pred_distribution: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_boxes: torch.Tensor,
        target_scores: torch.Tensor,
        foreground_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
            pred_distribution:
                [B, N, 4 * reg_max]

            pred_boxes:
                [B, N, 4]
                特征图坐标

            anchor_points:
                [N, 2]
                特征图坐标

            target_boxes:
                [B, N, 4]
                特征图坐标

            target_scores:
                [B, N, C]

            foreground_mask:
                [B, N]

            target_scores_sum:
                标量

        返回：
            iou_loss
            dfl_loss
        """

        if not foreground_mask.any():
            zero = pred_distribution.sum() * 0.0
            return zero, zero

        foreground_weights = target_scores.sum(
            dim=-1,
        )[foreground_mask]

        positive_pred_boxes = pred_boxes[
            foreground_mask
        ]

        positive_target_boxes = target_boxes[
            foreground_mask
        ]

        ciou = bbox_ciou(
            positive_pred_boxes,
            positive_target_boxes,
        )

        iou_loss = (
            (1.0 - ciou)
            * foreground_weights
        ).sum() / target_scores_sum

        # -----------------------------------------------
        # DFL 目标距离
        # -----------------------------------------------

        expanded_anchor_points = anchor_points.unsqueeze(0).expand(
            pred_boxes.shape[0],
            -1,
            -1,
        )

        target_distances = bbox2dist(
            expanded_anchor_points,
            target_boxes,
            reg_max=self.reg_max,
        )

        positive_pred_distribution = pred_distribution[
            foreground_mask
        ].reshape(
            -1,
            4,
            self.reg_max,
        )

        positive_target_distances = target_distances[
            foreground_mask
        ]

        dfl_loss = self.dfl_loss(
            positive_pred_distribution,
            positive_target_distances,
        )

        # 四条边取平均
        dfl_loss = dfl_loss.mean(dim=-1)

        dfl_loss = (
            dfl_loss
            * foreground_weights
        ).sum() / target_scores_sum

        return iou_loss, dfl_loss