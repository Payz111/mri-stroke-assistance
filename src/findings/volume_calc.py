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
    raise NotImplementedError()


def compute_max_diameter_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Compute the maximum Feret diameter of the lesion in mm.

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask.
    spacing:
        Voxel dimensions in mm ``(sx, sy, sz)``.

    Returns
    -------
    float
        Maximum diameter across any pair of lesion-boundary voxels.
    """
    raise NotImplementedError()
