"""Probability-map thresholding for binary mask generation."""

from __future__ import annotations

import numpy as np


def threshold_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Binarise a probability map at the given threshold.

    Parameters
    ----------
    prob_map:
        3-D array of predicted probabilities in [0, 1].
    threshold:
        Cut-off value; voxels >= *threshold* become foreground (1).

    Returns
    -------
    np.ndarray
        Binary mask of the same shape as *prob_map* (dtype ``uint8``).
    """
    raise NotImplementedError()
