"""DWI-FLAIR mismatch detection.

Estimates whether the ischaemic lesion shows a DWI-FLAIR mismatch,
which is a surrogate marker for acute onset (< 4.5 h) in stroke imaging.

Logic:
- DWI-positive + FLAIR-negative => mismatch positive (hyperacute, <4.5h)
- DWI-positive + FLAIR-positive => mismatch negative (>4.5h or subacute)
- Ambiguous signal levels => indeterminate
"""
from __future__ import annotations

import numpy as np


def detect_dwi_flair_mismatch(
    dwi: np.ndarray,
    flair: np.ndarray,
    lesion_mask: np.ndarray,
) -> tuple[str, float]:
    """Detect DWI-FLAIR mismatch within the lesion region.

    Compares the signal intensity within the lesion against the
    surrounding brain tissue. If DWI is bright but FLAIR is not
    significantly brighter than normal, the mismatch is positive.

    Parameters
    ----------
    dwi:
        DWI volume (co-registered, z-score normalised or raw).
    flair:
        FLAIR volume (co-registered, z-score normalised or raw).
    lesion_mask:
        Binary lesion mask.

    Returns
    -------
    tuple[str, float]
        ``(mismatch_status, score)`` where *mismatch_status* is one of
        ``"positive"``, ``"negative"``, or ``"indeterminate"``, and *score*
        is a continuous measure of the mismatch degree [0..1].
    """
    if lesion_mask.sum() == 0:
        return ("indeterminate", 0.0)

    lesion_bool = lesion_mask > 0
    brain_bool = (dwi > 0) | (flair > 0)
    normal_bool = brain_bool & ~lesion_bool

    # Need enough normal tissue to compute statistics
    if normal_bool.sum() < 100:
        return ("indeterminate", 0.0)

    # Compute z-scores of lesion relative to normal brain
    dwi_normal_mean = dwi[normal_bool].mean()
    dwi_normal_std = max(dwi[normal_bool].std(), 1e-7)
    dwi_lesion_z = (dwi[lesion_bool].mean() - dwi_normal_mean) / dwi_normal_std

    flair_normal_mean = flair[normal_bool].mean()
    flair_normal_std = max(flair[normal_bool].std(), 1e-7)
    flair_lesion_z = (flair[lesion_bool].mean() - flair_normal_mean) / flair_normal_std

    # DWI should be bright in lesion (z > 1.5 = clearly hyperintense)
    dwi_bright = dwi_lesion_z > 1.5

    # FLAIR hyperintensity threshold
    flair_bright = flair_lesion_z > 1.5
    flair_subtle = flair_lesion_z > 0.8

    if not dwi_bright:
        # DWI not convincingly bright -- uncertain
        return ("indeterminate", 0.3)

    # Mismatch score: high DWI z-score + low FLAIR z-score = mismatch
    # Normalise to [0..1]: score=1 means strong mismatch
    raw_score = (dwi_lesion_z - flair_lesion_z) / max(dwi_lesion_z, 1e-7)
    score = float(np.clip(raw_score, 0.0, 1.0))

    if dwi_bright and not flair_bright:
        if flair_subtle:
            return ("indeterminate", score)
        return ("positive", score)

    if dwi_bright and flair_bright:
        return ("negative", 1.0 - score)

    return ("indeterminate", 0.5)
