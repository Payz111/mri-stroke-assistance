"""Anatomic-location mapping of lesion masks.

Maps segmented lesion voxels to named brain regions using an optional
atlas volume.
"""
from __future__ import annotations

import numpy as np


def map_anatomic_location(
    mask: np.ndarray,
    atlas: np.ndarray | None = None,
) -> list[str]:
    """Identify the anatomic brain regions overlapped by the lesion.

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask in atlas space.
    atlas:
        Integer-labelled atlas volume (same shape as *mask*).
        If ``None``, a built-in atlas lookup table is used.

    Returns
    -------
    list[str]
        Region names sorted by descending overlap fraction.
    """
    raise NotImplementedError()
