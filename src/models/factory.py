"""Model registry and factory.

Centralises model construction so that the rest of the codebase only
needs a configuration dictionary to obtain a ready-to-use model.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn


def create_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate and return a segmentation model from *config*.

    Parameters
    ----------
    config:
        Must include at least ``model_name`` (e.g. ``"unet3d"``) and any
        model-specific hyper-parameters (``in_channels``, ``features``, …).

    Returns
    -------
    nn.Module
        An uninitialised (randomly weighted) model on CPU.

    Raises
    ------
    ValueError
        If the requested ``model_name`` is not in the registry.
    """
    raise NotImplementedError()
