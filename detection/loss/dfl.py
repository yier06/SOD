import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss。

    真实连续距离会被映射到相邻两个整数类别，
    然后分别计算加权交叉熵。
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()

        if reg_max <= 1:
            raise ValueError(
                "reg_max 必须大于 1"
            )

        self.reg_max = reg_max

    def forward(
        self,
        pred_distribution: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数：
            pred_distribution:
                [N, 4, reg_max]

            target:
                [N, 4]

        返回：
            [N, 4]
        """

        target = target.clamp(
            min=0,
            max=self.reg_max - 1 - 0.01,
        )

        target_left = target.long()
        target_right = target_left + 1

        weight_left = (
            target_right.to(target.dtype)
            - target
        )

        weight_right = (
            target
            - target_left.to(target.dtype)
        )

        prediction = pred_distribution.reshape(
            -1,
            self.reg_max,
        )

        target_left_flat = target_left.reshape(-1)
        target_right_flat = target_right.reshape(-1)

        left_loss = F.cross_entropy(
            prediction,
            target_left_flat,
            reduction="none",
        )

        right_loss = F.cross_entropy(
            prediction,
            target_right_flat,
            reduction="none",
        )

        loss = (
            left_loss * weight_left.reshape(-1)
            + right_loss * weight_right.reshape(-1)
        )

        return loss.reshape_as(target)