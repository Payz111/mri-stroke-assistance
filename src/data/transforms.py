"""MONAI transforms pipelines for training and validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from monai import transforms as T
from scipy.ndimage import zoom

from src.preprocess.intensity_norm import normalize_zscore


# Keys used throughout the pipeline
SPATIAL_KEYS = ["dwi", "adc", "flair", "mask"]
IMAGE_KEYS = ["dwi", "adc", "flair"]
LABEL_KEY = "mask"


class ResampleToReference(T.Transform):
    """Resample all modalities to match the reference key's spatial shape.

    Uses trilinear interpolation for images and nearest-neighbor for masks.
    """

    def __init__(self, reference_key: str = "dwi") -> None:
        self.reference_key = reference_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        ref_shape = d[self.reference_key].shape

        for key in IMAGE_KEYS:
            if d[key].shape != ref_shape:
                factors = [r / s for r, s in zip(ref_shape, d[key].shape)]
                d[key] = zoom(d[key], factors, order=1).astype(np.float32)

        # Resample mask with nearest-neighbor
        if d[LABEL_KEY].shape != ref_shape:
            factors = [r / s for r, s in zip(ref_shape, d[LABEL_KEY].shape)]
            d[LABEL_KEY] = zoom(d[LABEL_KEY], factors, order=0).astype(np.float32)

        return d


class NormalizePerModality(T.MapTransform):
    """Apply z-score normalization to each modality independently."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__(keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            d[key] = normalize_zscore(d[key]).astype(np.float32)
        return d


class StackModalities(T.Transform):
    """Stack DWI, ADC, FLAIR into a single (3, D, H, W) tensor."""

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        d["image"] = np.stack([d["dwi"], d["adc"], d["flair"]], axis=0)
        d["label"] = d["mask"][np.newaxis]  # (1, D, H, W)
        return d


def get_train_transforms(config: dict[str, Any] | None = None) -> T.Compose:
    """Build the augmentation + preprocessing pipeline for training."""
    return T.Compose([
        ResampleToReference(reference_key="dwi"),
        NormalizePerModality(keys=IMAGE_KEYS),
        StackModalities(),
        T.ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(config: dict[str, Any] | None = None) -> T.Compose:
    """Build the preprocessing pipeline for validation / inference."""
    return T.Compose([
        ResampleToReference(reference_key="dwi"),
        NormalizePerModality(keys=IMAGE_KEYS),
        StackModalities(),
        T.ToTensord(keys=["image", "label"]),
    ])
