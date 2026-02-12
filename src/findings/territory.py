"""Vascular-territory classification.

Determines which arterial territory a lesion most likely belongs to
(e.g. MCA, ACA, PCA, vertebrobasilar).
"""
from __future__ import annotations

import numpy as np


def classify_vascular_territory(
    mask: np.ndarray,
    atlas: np.ndarray | None = None,
) -> tuple[str, float]:
    """Classify the dominant vascular territory of the lesion.

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask in atlas space.
    atlas:
        Vascular-territory atlas volume. If ``None``, a built-in atlas
        is used.

    Returns
    -------
    tuple[str, float]
        ``(territory_name, confidence)`` where *confidence* is the
        fraction of lesion voxels falling in the dominant territory.
    """
    raise NotImplementedError()
