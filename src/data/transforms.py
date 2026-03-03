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
    After zoom, crops/pads to guarantee exact shape match (zoom can be off by
    +-1 voxel due to floating-point rounding).
    """

    def __init__(self, reference_key: str = "dwi") -> None:
        self.reference_key = reference_key

    @staticmethod
    def _force_shape(vol: np.ndarray, target: tuple) -> np.ndarray:
        """Crop or zero-pad *vol* so it exactly matches *target* shape."""
        if vol.shape == target:
            return vol
        result = np.zeros(target, dtype=vol.dtype)
        slices_src = []
        slices_dst = []
        for s, t in zip(vol.shape, target):
            m = min(s, t)
            slices_src.append(slice(0, m))
            slices_dst.append(slice(0, m))
        result[tuple(slices_dst)] = vol[tuple(slices_src)]
        return result

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        ref_shape = d[self.reference_key].shape

        for key in IMAGE_KEYS:
            if d[key].shape != ref_shape:
                factors = [r / s for r, s in zip(ref_shape, d[key].shape)]
                d[key] = zoom(d[key], factors, order=1).astype(np.float32)
                d[key] = self._force_shape(d[key], ref_shape)

        # Resample mask with nearest-neighbor
        if d[LABEL_KEY].shape != ref_shape:
            factors = [r / s for r, s in zip(ref_shape, d[LABEL_KEY].shape)]
            d[LABEL_KEY] = zoom(d[LABEL_KEY], factors, order=0).astype(np.float32)
            d[LABEL_KEY] = self._force_shape(d[LABEL_KEY], ref_shape)

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
    """Stack DWI, ADC, FLAIR into a single (3, D, H, W) tensor.

    Removes raw modality keys to avoid collation errors in DataLoader
    (raw volumes have different shapes across subjects).
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        d["image"] = np.stack([d["dwi"], d["adc"], d["flair"]], axis=0)
        d["label"] = d["mask"][np.newaxis]  # (1, D, H, W)
        # Remove raw keys -- only image/label needed downstream
        for key in ("dwi", "adc", "flair", "mask"):
            del d[key]
        return d


# Fixed spatial size so all subjects can be batched together.
# Chosen to cover most subjects (median ~112x112x73) with minimal padding.
SPATIAL_SIZE = (128, 128, 80)


def get_train_transforms(aug_config: dict[str, Any] | None = None) -> T.Compose:
    """Build the preprocessing pipeline for training.

    Parameters
    ----------
    aug_config:
        Augmentation config dict (reserved for future use).
        Augmentation experiments showed no improvement on ISLES 2022 (250 cases):
        baseline (no aug) dice=0.606 > v3 (flip+noise) 0.562 > v1 (full aug) 0.405.
    """
    spatial_size = SPATIAL_SIZE

    return T.Compose([
        ResampleToReference(reference_key="dwi"),
        NormalizePerModality(keys=IMAGE_KEYS),
        StackModalities(),
        T.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        T.CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
        T.ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(config: dict[str, Any] | None = None) -> T.Compose:
    """Build the preprocessing pipeline for validation / inference."""
    spatial_size = SPATIAL_SIZE
    if config and "spatial_size" in config:
        spatial_size = tuple(config["spatial_size"])

    return T.Compose([
        ResampleToReference(reference_key="dwi"),
        NormalizePerModality(keys=IMAGE_KEYS),
        StackModalities(),
        T.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
        T.CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
        T.ToTensord(keys=["image", "label"]),
    ])
