"""3-D UNet architecture for volumetric brain-lesion segmentation.

Uses MONAI's UNet implementation as the backbone -- battle-tested,
well-optimised, and standard in medical imaging research.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    """MONAI-based 3-D UNet with configurable channel progression.

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

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=strides,
            dropout=dropout,
            num_res_units=2,
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
