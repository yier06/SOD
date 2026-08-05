from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .iou import bbox_iou


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner。

    依据分类分数和 IoU 联合确定正样本。

    alignment_metric =
        class_score ** alpha
        * iou ** beta
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        super().__init__()

        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @staticmethod
    def select_candidates_in_gts(
        anchor_points: torch.Tensor,
        gt_boxes: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        判断候选点是否位于真实框内部。

        参数：
            anchor_points:
                [N, 2]，像素坐标

            gt_boxes:
                [B, M, 4]

        返回：
            [B, M, N]
        """

        batch_size, max_gt, _ = gt_boxes.shape
        num_anchors = anchor_points.shape[0]

        anchor_points = anchor_points.view(
            1,
            1,
            num_anchors,
            2,
        )

        gt_boxes = gt_boxes.view(
            batch_size,
            max_gt,
            1,
            4,
        )

        left = (
            anchor_points[..., 0]
            - gt_boxes[..., 0]
        )

        top = (
            anchor_points[..., 1]
            - gt_boxes[..., 1]
        )

        right = (
            gt_boxes[..., 2]
            - anchor_points[..., 0]
        )

        bottom = (
            gt_boxes[..., 3]
            - anchor_points[..., 1]
        )

        distances = torch.stack(
            [left, top, right, bottom],
            dim=-1,
        )

        return distances.amin(dim=-1) > eps

    def get_box_metrics(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算分类分数、IoU 和 alignment metric。

        参数：
            pred_scores:
                [B, N, C]，已经 sigmoid

            pred_boxes:
                [B, N, 4]

            gt_labels:
                [B, M]

            gt_boxes:
                [B, M, 4]

        返回：
            alignment_metrics:
                [B, M, N]

            overlaps:
                [B, M, N]
        """

        batch_size, num_anchors, _ = pred_scores.shape
        max_gt = gt_boxes.shape[1]

        expanded_pred_boxes = pred_boxes[:, None, :, :]
        expanded_gt_boxes = gt_boxes[:, :, None, :]

        overlaps = bbox_iou(
            expanded_pred_boxes,
            expanded_gt_boxes,
        ).clamp(min=0)

        gt_labels = gt_labels.long().clamp(
            min=0,
            max=self.num_classes - 1,
        )

        expanded_scores = pred_scores[:, None, :, :].expand(
            batch_size,
            max_gt,
            num_anchors,
            self.num_classes,
        )

        label_index = gt_labels[:, :, None, None].expand(
            batch_size,
            max_gt,
            num_anchors,
            1,
        )

        class_scores = torch.gather(
            expanded_scores,
            dim=3,
            index=label_index,
        ).squeeze(3)

        alignment_metrics = (
            class_scores.pow(self.alpha)
            * overlaps.pow(self.beta)
        )

        return alignment_metrics, overlaps

    def select_topk_candidates(
        self,
        metrics: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        对每个真实框选择 alignment metric 最大的 top-k 候选点。

        参数：
            metrics:
                [B, M, N]

            valid_mask:
                [B, M, N]

        返回：
            [B, M, N]
        """

        num_anchors = metrics.shape[-1]
        topk = min(self.topk, num_anchors)

        masked_metrics = metrics.masked_fill(
            ~valid_mask,
            -1,
        )

        topk_metrics, topk_indices = torch.topk(
            masked_metrics,
            k=topk,
            dim=-1,
            largest=True,
        )

        topk_valid = topk_metrics > 0

        topk_mask = torch.zeros_like(
            metrics,
            dtype=torch.bool,
        )

        topk_mask.scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_valid,
        )

        return topk_mask

    @staticmethod
    def resolve_multi_assignments(
        positive_mask: torch.Tensor,
        overlaps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        一个 anchor 如果同时匹配多个 GT，
        则选择 IoU 最大的 GT。

        参数：
            positive_mask:
                [B, M, N]

            overlaps:
                [B, M, N]

        返回：
            target_gt_index:
                [B, N]

            foreground_mask:
                [B, N]

            final_positive_mask:
                [B, M, N]
        """

        foreground_count = positive_mask.sum(dim=1)
        multi_mask = foreground_count > 1

        if multi_mask.any():
            max_overlap_gt = overlaps.argmax(dim=1)

            best_gt_mask = F.one_hot(
                max_overlap_gt,
                num_classes=positive_mask.shape[1],
            ).permute(0, 2, 1).bool()

            positive_mask = torch.where(
                multi_mask[:, None, :],
                best_gt_mask,
                positive_mask,
            )

        foreground_mask = positive_mask.any(dim=1)
        target_gt_index = positive_mask.float().argmax(dim=1)

        return (
            target_gt_index,
            foreground_mask,
            positive_mask,
        )

    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ):
        """
        参数：
            pred_scores:
                [B, N, C]，Sigmoid 后分数

            pred_boxes:
                [B, N, 4]，像素坐标

            anchor_points:
                [N, 2]，像素坐标

            gt_labels:
                [B, M]

            gt_boxes:
                [B, M, 4]

            mask_gt:
                [B, M]

        返回：
            target_labels:
                [B, N]

            target_boxes:
                [B, N, 4]

            target_scores:
                [B, N, C]

            foreground_mask:
                [B, N]

            target_gt_index:
                [B, N]
        """

        batch_size, num_anchors, _ = pred_scores.shape
        max_gt = gt_boxes.shape[1]

        if max_gt == 0:
            return self._empty_targets(
                batch_size=batch_size,
                num_anchors=num_anchors,
                device=pred_scores.device,
                dtype=pred_scores.dtype,
            )

        candidate_mask = self.select_candidates_in_gts(
            anchor_points,
            gt_boxes,
            eps=self.eps,
        )

        valid_gt_mask = mask_gt[:, :, None].bool()

        candidate_mask = (
            candidate_mask
            & valid_gt_mask
        )

        alignment_metrics, overlaps = self.get_box_metrics(
            pred_scores,
            pred_boxes,
            gt_labels,
            gt_boxes,
        )

        alignment_metrics = alignment_metrics * candidate_mask
        overlaps = overlaps * candidate_mask

        topk_mask = self.select_topk_candidates(
            alignment_metrics,
            candidate_mask,
        )

        positive_mask = (
            topk_mask
            & candidate_mask
            & valid_gt_mask
        )

        (
            target_gt_index,
            foreground_mask,
            positive_mask,
        ) = self.resolve_multi_assignments(
            positive_mask,
            overlaps,
        )

        batch_indices = torch.arange(
            batch_size,
            device=pred_scores.device,
        )[:, None]

        target_labels = gt_labels[
            batch_indices,
            target_gt_index,
        ].long()

        target_boxes = gt_boxes[
            batch_indices,
            target_gt_index,
        ]

        target_labels = target_labels.clamp(
            min=0,
            max=self.num_classes - 1,
        )

        target_scores = F.one_hot(
            target_labels,
            num_classes=self.num_classes,
        ).to(pred_scores.dtype)

        target_scores = (
            target_scores
            * foreground_mask.unsqueeze(-1)
        )

        # -----------------------------------------------
        # 使用 alignment metric 作为软标签质量分数
        # -----------------------------------------------

        positive_alignment = (
            alignment_metrics * positive_mask
        )

        positive_overlaps = (
            overlaps * positive_mask
        )

        max_alignment_per_gt = positive_alignment.amax(
            dim=-1,
            keepdim=True,
        )

        max_overlap_per_gt = positive_overlaps.amax(
            dim=-1,
            keepdim=True,
        )

        normalized_alignment = (
            positive_alignment
            * max_overlap_per_gt
            / (
                max_alignment_per_gt
                + self.eps
            )
        )

        anchor_quality = normalized_alignment.amax(dim=1)

        target_scores = (
            target_scores
            * anchor_quality.unsqueeze(-1)
        )

        return (
            target_labels,
            target_boxes,
            target_scores,
            foreground_mask,
            target_gt_index,
        )

    def _empty_targets(
        self,
        batch_size: int,
        num_anchors: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        target_labels = torch.zeros(
            batch_size,
            num_anchors,
            dtype=torch.long,
            device=device,
        )

        target_boxes = torch.zeros(
            batch_size,
            num_anchors,
            4,
            dtype=dtype,
            device=device,
        )

        target_scores = torch.zeros(
            batch_size,
            num_anchors,
            self.num_classes,
            dtype=dtype,
            device=device,
        )

        foreground_mask = torch.zeros(
            batch_size,
            num_anchors,
            dtype=torch.bool,
            device=device,
        )

        target_gt_index = torch.zeros(
            batch_size,
            num_anchors,
            dtype=torch.long,
            device=device,
        )

        return (
            target_labels,
            target_boxes,
            target_scores,
            foreground_mask,
            target_gt_index,
        )