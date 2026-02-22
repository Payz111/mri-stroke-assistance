"""Intensity normalisation routines for MRI volumes."""

from __future__ import annotations

import numpy as np


def normalize_zscore(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Z-score normalise *volume*, optionally within *mask*.

    Mean and std are computed over the masked region (or entire volume),
    then the full volume is normalised.
    """
    if mask is not None:
        roi = volume[mask > 0]
    else:
        roi = volume[volume > 0]  # exclude background zeros

    mean = roi.mean()
    std = roi.std()

    if std < 1e-8:
        return volume - mean

    return (volume - mean) / std


def normalize_percentile(
    volume: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    """Clip and rescale *volume* to the [lower, upper] percentile range.

    Returns volume scaled to [0, 1].
    """
    p_low = np.percentile(volume, lower)
    p_high = np.percentile(volume, upper)

    volume = np.clip(volume, p_low, p_high)

    denom = p_high - p_low
    if denom < 1e-8:
        return np.zeros_like(volume)

    return (volume - p_low) / denom
