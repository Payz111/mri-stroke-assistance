"""3-D UNet architecture for volumetric brain-lesion segmentation.

Implements an encoder-decoder UNet with skip connections operating on
3-D tensors of shape ``(B, C, D, H, W)``.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    """Standard 3-D UNet with configurable channel progression.

    Parameters
    ----------
    in_channels:
        Number of input channels (e.g. 3 for DWI + ADC + FLAIR).
    out_channels:
        Number of output channels / classes (e.g. 1 for binary mask).
    features:
        Sequence of feature-map sizes for each encoder level, e.g.
        ``(32, 64, 128, 256)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, in_channels, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape ``(B, out_channels, D, H, W)``.
        """
        raise NotImplementedError()
