"""MONAI transforms pipelines for training and validation.

Provides factory functions that build ``monai.transforms.Compose`` objects
from a configuration dictionary, keeping augmentation logic centralised.
"""
from __future__ import annotations

from typing import Any

import monai.transforms


def get_train_transforms(config: dict[str, Any]) -> monai.transforms.Compose:
    """Build the augmentation + preprocessing pipeline for training.

    Parameters
    ----------
    config:
        Configuration dictionary with keys such as ``patch_size``,
        ``intensity_aug``, ``spatial_aug``, etc.

    Returns
    -------
    monai.transforms.Compose
        Composed transform ready to be passed to the Dataset.
    """
    raise NotImplementedError()


def get_val_transforms(config: dict[str, Any]) -> monai.transforms.Compose:
    """Build the preprocessing pipeline for validation / inference.

    Parameters
    ----------
    config:
        Same configuration dictionary used for training (augmentation
        flags are ignored).

    Returns
    -------
    monai.transforms.Compose
        Composed transform ready to be passed to the Dataset.
    """
    raise NotImplementedError()
