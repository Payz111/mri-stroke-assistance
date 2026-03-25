"""Model and loss registry and factory."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.attention_unet import AttentionUNet3D
from src.models.losses import DiceFocalLoss, DiceLoss, FocalLoss
from src.models.unet3d import UNet3D

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "unet3d": UNet3D,
    "attention_unet3d": AttentionUNet3D,
}

LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "dice": DiceLoss,
    "focal": FocalLoss,
    "dice_focal": DiceFocalLoss,
}


def create_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate a segmentation model from *config*.

    Expected config keys: name, in_channels, out_channels, features, dropout.
    """
    name = config.get("name", "unet3d")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY)}")

    cls = MODEL_REGISTRY[name]
    return cls(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 1),
        features=config.get("features", [32, 64, 128, 256]),
        dropout=config.get("dropout", 0.1),
    )


def create_loss(config: dict[str, Any]) -> nn.Module:
    """Instantiate a loss function from *config*.

    Expected config keys: type, dice_weight, focal_weight, focal_gamma.
    """
    loss_type = config.get("type", "dice_focal")
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {loss_type!r}. Available: {list(LOSS_REGISTRY)}")

    if loss_type == "dice_focal":
        return DiceFocalLoss(
            dice_weight=config.get("dice_weight", 0.5),
            focal_weight=config.get("focal_weight", 0.5),
            focal_gamma=config.get("focal_gamma", 2.0),
        )
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=config.get("focal_gamma", 2.0))

    return LOSS_REGISTRY[loss_type]()
