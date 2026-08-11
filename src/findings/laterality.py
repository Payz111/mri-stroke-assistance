"""Laterality detection -- determine which hemisphere(s) a lesion occupies."""

from __future__ import annotations

import numpy as np


def detect_laterality(
    mask: np.ndarray,
    midline_idx: int | None = None,
) -> str:
    """Determine the hemispheric laterality of the lesion.

    The first spatial axis (axis 0, typically Left-Right in RAS/LAS) is
    used to split the volume at the midline.

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask.
    midline_idx:
        Index along the left-right axis that separates the hemispheres.
        If ``None``, the midline is estimated as the centre of the volume.

    Returns
    -------
    str
        One of ``"left"``, ``"right"``, or ``"bilateral"``.
    """
    if mask.sum() == 0:
        return "bilateral"

    if midline_idx is None:
        midline_idx = mask.shape[0] // 2

    left_count = int((mask[:midline_idx] > 0).sum())
    right_count = int((mask[midline_idx:] > 0).sum())
    total = left_count + right_count

    if total == 0:
        return "bilateral"

    left_frac = left_count / total
    right_frac = right_count / total

    # If >80% of voxels on one side -> unilateral
    if left_frac > 0.8:
        return "left"
    if right_frac > 0.8:
        return "right"
    return "bilateral"
