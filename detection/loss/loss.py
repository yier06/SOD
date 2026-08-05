from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor import dist2bbox, make_anchors
from .assigner import TaskAlignedAssigner
from .bbox_loss import BboxLoss


class YOLO11DetectionLoss(nn.Module):
    """
    YOLO11 风格目标检测损失。

    总损失：

        total_loss =
            box_weight * box_loss
            + cls_weight * cls_loss
            + dfl_weight * dfl_loss

    模型输出格式：

        [
            {
                "regression": [B, 4 * reg_max, H3, W3],
                "classification": [B, C, H3, W3]
            },
            {
                "regression": [B, 4 * reg_max, H4, W4],
                "classification": [B, C, H4, W4]
            },
            {
                "regression": [B, 4 * reg_max, H5, W5],
                "classification": [B, C, H5, W5]
            }
        ]

    标注支持两种格式：

    格式一：

        targets = [
            {
                "boxes": Tensor[N1, 4],
                "labels": Tensor[N1]
            },
            ...
        ]

    格式二：

        targets = {
            "boxes": Tensor[N, 4],
            "labels": Tensor[N],
            "batch_idx": Tensor[N]
        }

    boxes 默认要求：
        xyxy
        绝对像素坐标
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        strides: Sequence[int] = (8, 16, 32),
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        assigner_topk: int = 10,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes 必须大于 0"
            )

        if reg_max <= 1:
            raise ValueError(
                "reg_max 必须大于 1"
            )

        if len(strides) != 3:
            raise ValueError(
                "当前实现要求三个检测尺度"
            )

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = tuple(strides)

        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

        self.assigner = TaskAlignedAssigner(
            topk=assigner_topk,
            num_classes=num_classes,
            alpha=assigner_alpha,
            beta=assigner_beta,
        )

        self.bbox_loss = BboxLoss(
            reg_max=reg_max,
        )

        projection = torch.arange(
            reg_max,
            dtype=torch.float32,
        )

        self.register_buffer(
            "projection",
            projection,
            persistent=False,
        )

    def flatten_predictions(
        self,
        raw_outputs: List[Dict[str, torch.Tensor]],
    ):
        """
        将三个尺度的预测展平并拼接。
        """

        pred_distributions = []
        pred_scores = []
        feature_shapes = []

        batch_size = None

        for level, output in enumerate(raw_outputs):

            if not isinstance(output, dict):
                raise TypeError(
                    f"第 {level} 个输出必须是字典"
                )

            if "regression" not in output:
                raise KeyError(
                    f"第 {level} 个输出缺少 regression"
                )

            if "classification" not in output:
                raise KeyError(
                    f"第 {level} 个输出缺少 classification"
                )

            regression = output["regression"]
            classification = output["classification"]

            if regression.ndim != 4:
                raise ValueError(
                    "regression 必须为 [B, C, H, W]"
                )

            if classification.ndim != 4:
                raise ValueError(
                    "classification 必须为 [B, C, H, W]"
                )

            current_batch_size = regression.shape[0]

            if batch_size is None:
                batch_size = current_batch_size
            elif batch_size != current_batch_size:
                raise ValueError(
                    "不同尺度的 batch size 不一致"
                )

            _, regression_channels, height, width = (
                regression.shape
            )

            expected_regression_channels = (
                4 * self.reg_max
            )

            if (
                regression_channels
                != expected_regression_channels
            ):
                raise ValueError(
                    "回归通道数错误："
                    f"期望 {expected_regression_channels}，"
                    f"实际 {regression_channels}"
                )

            if (
                classification.shape[1]
                != self.num_classes
            ):
                raise ValueError(
                    "分类通道数错误："
                    f"期望 {self.num_classes}，"
                    f"实际 {classification.shape[1]}"
                )

            feature_shapes.append(
                (height, width)
            )

            regression = regression.permute(
                0,
                2,
                3,
                1,
            ).contiguous()

            regression = regression.reshape(
                current_batch_size,
                height * width,
                4 * self.reg_max,
            )

            classification = classification.permute(
                0,
                2,
                3,
                1,
            ).contiguous()

            classification = classification.reshape(
                current_batch_size,
                height * width,
                self.num_classes,
            )

            pred_distributions.append(regression)
            pred_scores.append(classification)

        pred_distributions = torch.cat(
            pred_distributions,
            dim=1,
        )

        pred_scores = torch.cat(
            pred_scores,
            dim=1,
        )

        return (
            pred_distributions,
            pred_scores,
            feature_shapes,
        )

    def decode_boxes(
        self,
        pred_distribution: torch.Tensor,
        anchor_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        DFL 解码。

        参数：
            pred_distribution:
                [B, N, 4 * reg_max]

            anchor_points:
                [N, 2]

        返回：
            [B, N, 4]
            特征图坐标
        """

        batch_size, num_anchors, _ = (
            pred_distribution.shape
        )

        distribution = pred_distribution.reshape(
            batch_size,
            num_anchors,
            4,
            self.reg_max,
        )

        probability = distribution.softmax(dim=-1)

        projection = self.projection.to(
            device=probability.device,
            dtype=probability.dtype,
        )

        distances = (
            probability
            * projection.view(1, 1, 1, -1)
        ).sum(dim=-1)

        return dist2bbox(
            distances,
            anchor_points,
        )

    def preprocess_targets(
        self,
        targets: Any,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        将不同格式标注整理为：

            gt_labels: [B, M]
            gt_boxes:  [B, M, 4]
            mask_gt:   [B, M]
        """

        per_image_boxes: List[torch.Tensor] = []
        per_image_labels: List[torch.Tensor] = []

        # -----------------------------------------------
        # 格式一：字典列表
        # -----------------------------------------------

        if isinstance(targets, (list, tuple)):

            if len(targets) != batch_size:
                raise ValueError(
                    "targets 列表长度必须等于 batch size"
                )

            for target in targets:

                if not isinstance(target, dict):
                    raise TypeError(
                        "targets 列表中的元素必须是字典"
                    )

                boxes = target.get("boxes")
                labels = target.get("labels")

                if boxes is None or labels is None:
                    raise KeyError(
                        "每个 target 必须包含 "
                        "boxes 和 labels"
                    )

                boxes = boxes.to(
                    device=device,
                    dtype=dtype,
                )

                labels = labels.to(
                    device=device,
                    dtype=torch.long,
                )

                self.validate_target_shapes(
                    boxes,
                    labels,
                )

                per_image_boxes.append(boxes)
                per_image_labels.append(labels)

        # -----------------------------------------------
        # 格式二：合并字典
        # -----------------------------------------------

        elif isinstance(targets, dict):

            boxes = targets.get("boxes")
            labels = targets.get("labels")
            batch_idx = targets.get("batch_idx")

            if (
                boxes is None
                or labels is None
                or batch_idx is None
            ):
                raise KeyError(
                    "合并 targets 必须包含："
                    "boxes、labels、batch_idx"
                )

            boxes = boxes.to(
                device=device,
                dtype=dtype,
            )

            labels = labels.to(
                device=device,
                dtype=torch.long,
            )

            batch_idx = batch_idx.to(
                device=device,
                dtype=torch.long,
            )

            self.validate_target_shapes(
                boxes,
                labels,
            )

            if batch_idx.ndim != 1:
                raise ValueError(
                    "batch_idx 必须是一维张量"
                )

            if batch_idx.shape[0] != boxes.shape[0]:
                raise ValueError(
                    "batch_idx 数量必须和 boxes 数量一致"
                )

            for image_index in range(batch_size):

                image_mask = (
                    batch_idx == image_index
                )

                per_image_boxes.append(
                    boxes[image_mask]
                )

                per_image_labels.append(
                    labels[image_mask]
                )

        else:
            raise TypeError(
                "targets 必须是字典或字典列表"
            )

        max_gt = max(
            (
                boxes.shape[0]
                for boxes in per_image_boxes
            ),
            default=0,
        )

        # 没有任何 GT 时仍保留一列，避免部分算子处理空维度
        padded_max_gt = max(max_gt, 1)

        gt_boxes = torch.zeros(
            batch_size,
            padded_max_gt,
            4,
            device=device,
            dtype=dtype,
        )

        gt_labels = torch.zeros(
            batch_size,
            padded_max_gt,
            device=device,
            dtype=torch.long,
        )

        mask_gt = torch.zeros(
            batch_size,
            padded_max_gt,
            device=device,
            dtype=torch.bool,
        )

        for image_index, (
            boxes,
            labels,
        ) in enumerate(
            zip(
                per_image_boxes,
                per_image_labels,
            )
        ):
            num_gt = boxes.shape[0]

            if num_gt == 0:
                continue

            gt_boxes[
                image_index,
                :num_gt,
            ] = boxes

            gt_labels[
                image_index,
                :num_gt,
            ] = labels

            mask_gt[
                image_index,
                :num_gt,
            ] = True

        return (
            gt_labels,
            gt_boxes,
            mask_gt,
        )

    def validate_target_shapes(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        检查标注形状及合法性。
        """

        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(
                "boxes 必须为 [N, 4]"
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels 必须为 [N]"
            )

        if boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                "boxes 和 labels 数量不一致"
            )

        if labels.numel() > 0:
            if labels.min() < 0:
                raise ValueError(
                    "类别编号不能小于 0"
                )

            if labels.max() >= self.num_classes:
                raise ValueError(
                    "类别编号超出 num_classes 范围"
                )

        if boxes.numel() > 0:
            invalid_width = (
                boxes[:, 2] <= boxes[:, 0]
            )

            invalid_height = (
                boxes[:, 3] <= boxes[:, 1]
            )

            if (
                invalid_width.any()
                or invalid_height.any()
            ):
                raise ValueError(
                    "存在非法 xyxy 边界框，"
                    "必须满足 x2>x1 且 y2>y1"
                )

    def forward(
        self,
        raw_outputs: List[Dict[str, torch.Tensor]],
        targets: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        计算 YOLO11 检测损失。
        """

        if len(raw_outputs) != len(self.strides):
            raise ValueError(
                "模型输出尺度数量与 strides 不一致"
            )

        (
            pred_distributions,
            pred_logits,
            feature_shapes,
        ) = self.flatten_predictions(raw_outputs)

        device = pred_logits.device
        dtype = pred_logits.dtype
        batch_size = pred_logits.shape[0]

        # -----------------------------------------------
        # 生成 anchor points
        # -----------------------------------------------

        anchor_points, stride_tensor = make_anchors(
            feature_shapes=feature_shapes,
            strides=self.strides,
            device=device,
            dtype=dtype,
        )

        # -----------------------------------------------
        # 解码预测框
        # -----------------------------------------------

        pred_boxes_feature = self.decode_boxes(
            pred_distributions,
            anchor_points,
        )

        pred_boxes_pixel = (
            pred_boxes_feature
            * stride_tensor.unsqueeze(0)
        )

        anchor_points_pixel = (
            anchor_points
            * stride_tensor
        )

        # -----------------------------------------------
        # 整理标签
        # -----------------------------------------------

        (
            gt_labels,
            gt_boxes_pixel,
            mask_gt,
        ) = self.preprocess_targets(
            targets=targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # -----------------------------------------------
        # Task-Aligned Assigner
        # detach 防止标签分配参与梯度计算
        # -----------------------------------------------

        with torch.no_grad():
            (
                target_labels,
                target_boxes_pixel,
                target_scores,
                foreground_mask,
                target_gt_index,
            ) = self.assigner(
                pred_scores=pred_logits.detach().sigmoid(),
                pred_boxes=pred_boxes_pixel.detach(),
                anchor_points=anchor_points_pixel,
                gt_labels=gt_labels,
                gt_boxes=gt_boxes_pixel,
                mask_gt=mask_gt,
            )

        target_scores_sum = target_scores.sum().clamp(
            min=1.0
        )

        # -----------------------------------------------
        # 分类损失
        # -----------------------------------------------

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            target_scores,
            reduction="sum",
        )

        cls_loss = cls_loss / target_scores_sum

        # -----------------------------------------------
        # 将目标框转换回各自特征层坐标
        # -----------------------------------------------

        target_boxes_feature = (
            target_boxes_pixel
            / stride_tensor.unsqueeze(0)
        )

        # -----------------------------------------------
        # 边界框和 DFL 损失
        # -----------------------------------------------

        box_loss, dfl_loss = self.bbox_loss(
            pred_distribution=pred_distributions,
            pred_boxes=pred_boxes_feature,
            anchor_points=anchor_points,
            target_boxes=target_boxes_feature,
            target_scores=target_scores,
            foreground_mask=foreground_mask,
            target_scores_sum=target_scores_sum,
        )

        weighted_box_loss = (
            box_loss * self.box_weight
        )

        weighted_cls_loss = (
            cls_loss * self.cls_weight
        )

        weighted_dfl_loss = (
            dfl_loss * self.dfl_weight
        )

        total_loss = (
            weighted_box_loss
            + weighted_cls_loss
            + weighted_dfl_loss
        )

        return {
            "loss": total_loss,
            "box_loss": weighted_box_loss,
            "cls_loss": weighted_cls_loss,
            "dfl_loss": weighted_dfl_loss,
            "raw_box_loss": box_loss.detach(),
            "raw_cls_loss": cls_loss.detach(),
            "raw_dfl_loss": dfl_loss.detach(),
            "num_foreground": foreground_mask.sum().detach(),
        }