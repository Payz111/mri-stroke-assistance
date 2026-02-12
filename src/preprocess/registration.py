"""Co-registration of multi-modal MRI volumes.

Provides rigid (and optionally affine) registration so that all
modalities share the same coordinate frame before segmentation.
"""
from __future__ import annotations

import numpy as np


def coregister(
    moving: np.ndarray,
    fixed: np.ndarray,
    method: str = "rigid",
) -> np.ndarray:
    """Register *moving* volume to *fixed* reference space.

    Parameters
    ----------
    moving:
        The volume to be transformed (e.g. FLAIR).
    fixed:
        The reference volume (e.g. DWI).
    method:
        Registration strategy — ``"rigid"`` or ``"affine"``.

    Returns
    -------
    np.ndarray
        The *moving* volume resampled into *fixed* space.
    """
    raise NotImplementedError()
