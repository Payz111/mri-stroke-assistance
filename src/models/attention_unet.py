"""3-D Attention U-Net for volumetric brain-lesion segmentation.

Uses MONAI's AttentionUnet -- adds attention gates at each skip connection
to help the model focus on small lesion regions.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet


class AttentionUNet3D(nn.Module):
    """MONAI-based 3-D Attention U-Net.

    Parameters
    ----------
    in_channels:
        Number of input channels (e.g. 3 for DWI + ADC + FLAIR).
    out_channels:
        Number of output channels / classes (e.g. 1 for binary mask).
    features:
        Sequence of feature-map sizes for each encoder level.
    dropout:
        Dropout probability applied after each conv block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        features = tuple(features)
        strides = tuple(2 for _ in range(len(features) - 1))

        self.net = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides,
            dropout=dropout,
        )

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
        return self.net(x)
