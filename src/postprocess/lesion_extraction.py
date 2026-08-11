"""Per-lesion statistics extraction from a segmented mask."""

from __future__ import annotations

import numpy as np


def extract_lesion_stats(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> list[dict]:
    """Compute per-lesion statistics from a labelled mask.

    Parameters
    ----------
    mask:
        3-D labelled mask where each connected component has a unique
        positive integer ID (background = 0).
    spacing:
        Voxel dimensions in mm ``(sx, sy, sz)``.

    Returns
    -------
    list[dict]
        One dictionary per lesion containing keys such as ``label``,
        ``volume_ml``, ``centroid``, ``bounding_box``, etc.
    """
    raise NotImplementedError()
