"""Loss functions for ischemic-stroke lesion segmentation.

Provides Dice loss, Focal loss, and a combined Dice + Focal variant
commonly used for highly imbalanced medical-image segmentation tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Parameters
    ----------
    smooth:
        Laplace smoothing term to avoid division by zero.
    sigmoid:
        If True, apply sigmoid to predictions before computing loss.
    """

    def __init__(self, smooth: float = 1.0, sigmoid: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # Flatten spatial dims: (B, C, ...) -> (B, C, N)
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss to down-weight easy examples.

    Parameters
    ----------
    alpha:
        Balancing factor for the positive class.
    gamma:
        Focusing parameter; higher values increase focus on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.exp(-bce)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceFocalLoss(nn.Module):
    """Weighted combination of Dice and Focal losses.

    Parameters
    ----------
    dice_weight:
        Scalar multiplier for the Dice term.
    focal_weight:
        Scalar multiplier for the Focal term.
    focal_gamma:
        Gamma for the Focal loss component.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(sigmoid=True)
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice_loss(pred, target)
            + self.focal_weight * self.focal_loss(pred, target)
        )
