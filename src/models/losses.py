"""Loss functions for ischemic-stroke lesion segmentation.

Provides Dice loss, Focal loss, and a combined Dice + Focal variant
commonly used for highly imbalanced medical-image segmentation tasks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary or multi-class segmentation.

    Parameters
    ----------
    smooth:
        Laplace smoothing term to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        raise NotImplementedError()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss.

        Parameters
        ----------
        pred:
            Predicted probabilities ``(B, C, ...)``.
        target:
            Ground-truth labels ``(B, C, ...)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        raise NotImplementedError()


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
        raise NotImplementedError()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Focal loss.

        Parameters
        ----------
        pred:
            Predicted logits ``(B, C, ...)``.
        target:
            Ground-truth labels ``(B, C, ...)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        raise NotImplementedError()


class DiceFocalLoss(nn.Module):
    """Weighted combination of Dice and Focal losses.

    Parameters
    ----------
    dice_weight:
        Scalar multiplier for the Dice term.
    focal_weight:
        Scalar multiplier for the Focal term.
    """

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0) -> None:
        super().__init__()
        raise NotImplementedError()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined Dice + Focal loss.

        Parameters
        ----------
        pred:
            Predicted logits / probabilities ``(B, C, ...)``.
        target:
            Ground-truth labels ``(B, C, ...)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        raise NotImplementedError()
