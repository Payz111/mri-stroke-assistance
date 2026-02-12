"""Laterality detection — determine which hemisphere(s) a lesion occupies."""
from __future__ import annotations

import numpy as np


def detect_laterality(
    mask: np.ndarray,
    midline_idx: int | None = None,
) -> str:
    """Determine the hemispheric laterality of the lesion.

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
    raise NotImplementedError()
