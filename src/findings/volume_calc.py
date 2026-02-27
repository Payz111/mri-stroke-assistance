"""Lesion volume and size measurements.

Converts voxel counts into physical volume (mL) and computes the
maximum lesion diameter in mm.
"""
from __future__ import annotations

import numpy as np


def compute_volume_ml(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Compute the total lesion volume in millilitres.

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask.
    spacing:
        Voxel dimensions in mm ``(sx, sy, sz)``.

    Returns
    -------
    float
        Total lesion volume in mL.
    """
    voxel_vol_mm3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    n_voxels = int((mask > 0).sum())
    return n_voxels * voxel_vol_mm3 / 1000.0


def compute_max_diameter_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Compute the maximum Feret diameter of the lesion in mm.

    Uses the bounding box diagonal as a fast approximation.
    For small lesions this is close to the true Feret diameter.

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask.
    spacing:
        Voxel dimensions in mm ``(sx, sy, sz)``.

    Returns
    -------
    float
        Maximum diameter across the lesion bounding box (mm).
        Returns 0.0 if the mask is empty.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0.0

    # Physical coordinates
    phys = coords.astype(np.float64) * np.array(spacing, dtype=np.float64)

    # Bounding-box diagonal
    mins = phys.min(axis=0)
    maxs = phys.max(axis=0)
    diameter = float(np.linalg.norm(maxs - mins))
    return diameter
