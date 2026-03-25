"""Test-time augmentation (TTA) for segmentation models.

Simple flip-based TTA: average predictions over the original and
flipped versions of the input to improve robustness.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def predict_with_tta(
    model: nn.Module,
    image: torch.Tensor,
    flip_axes: list[int] | None = None,
) -> torch.Tensor:
    """Run inference with test-time augmentation.

    Parameters
    ----------
    model:
        Segmentation model (expects sigmoid output or logits).
    image:
        Input tensor of shape ``(1, C, D, H, W)``.
    flip_axes:
        Spatial axes to flip (2=left-right, 3=anterior-posterior, 4=sup-inf).
        Default: [2] (left-right flip only).

    Returns
    -------
    torch.Tensor
        Averaged probability map of shape ``(1, 1, D, H, W)``.
    """
    if flip_axes is None:
        flip_axes = [2]

    with torch.no_grad():
        # Original prediction
        logits = model(image)
        prob_sum = torch.sigmoid(logits)
        n = 1

        # Flipped predictions
        for axis in flip_axes:
            flipped = torch.flip(image, dims=[axis])
            flipped_logits = model(flipped)
            flipped_prob = torch.sigmoid(flipped_logits)
            # Flip back to original orientation
            prob_sum = prob_sum + torch.flip(flipped_prob, dims=[axis])
            n += 1

    return prob_sum / n
