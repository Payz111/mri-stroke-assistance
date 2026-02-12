"""DWI-FLAIR mismatch detection.

Estimates whether the ischaemic lesion shows a DWI-FLAIR mismatch,
which is a surrogate marker for acute onset (< 4.5 h) in stroke imaging.
"""
from __future__ import annotations

import numpy as np


def detect_dwi_flair_mismatch(
    dwi: np.ndarray,
    flair: np.ndarray,
    lesion_mask: np.ndarray,
) -> tuple[str, float]:
    """Detect DWI-FLAIR mismatch within the lesion region.

    Parameters
    ----------
    dwi:
        DWI volume (co-registered).
    flair:
        FLAIR volume (co-registered).
    lesion_mask:
        Binary lesion mask.

    Returns
    -------
    tuple[str, float]
        ``(mismatch_status, score)`` where *mismatch_status* is one of
        ``"positive"``, ``"negative"``, or ``"indeterminate"``, and *score*
        is a continuous measure of the mismatch degree.
    """
    raise NotImplementedError()
