"""Post-processing utilities for segmentation predictions."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 10,
) -> np.ndarray:
    """Remove connected components smaller than *min_size* voxels.

    Parameters
    ----------
    mask:
        Binary mask (3D array).
    min_size:
        Minimum number of voxels for a component to be kept.

    Returns
    -------
    np.ndarray
        Cleaned binary mask with small components removed.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary.astype(np.float32)

    labeled, n_components = label(binary)
    cleaned = np.zeros_like(binary, dtype=np.float32)

    for i in range(1, n_components + 1):
        component = labeled == i
        if component.sum() >= min_size:
            cleaned[component] = 1.0

    return cleaned
