"""Intensity normalisation routines for MRI volumes.

Provides z-score and percentile-based normalisation, optionally
restricted to a foreground mask.
"""
from __future__ import annotations

import numpy as np


def normalize_zscore(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Z-score normalise *volume*, optionally within *mask*.

    Parameters
    ----------
    volume:
        3-D MRI intensity array.
    mask:
        Optional binary mask; if provided, mean and std are computed
        only over the masked region but the entire volume is normalised.

    Returns
    -------
    np.ndarray
        Normalised volume with zero mean and unit variance (within mask).
    """
    raise NotImplementedError()


def normalize_percentile(
    volume: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    """Clip and rescale *volume* to the [lower, upper] percentile range.

    Parameters
    ----------
    volume:
        3-D MRI intensity array.
    lower:
        Lower percentile for clipping (default 1.0).
    upper:
        Upper percentile for clipping (default 99.0).

    Returns
    -------
    np.ndarray
        Volume scaled to the [0, 1] range after percentile clipping.
    """
    raise NotImplementedError()
