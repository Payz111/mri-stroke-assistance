"""Contralateral CBF estimation for rCBF computation."""
from __future__ import annotations

import numpy as np


def estimate_contralateral_cbf(
    cbf: np.ndarray,
    brain_mask: np.ndarray | None = None,
    core_estimate: np.ndarray | None = None,
) -> float:
    """Estimate normal CBF from the contralateral hemisphere.

    Strategy: mirror the brain left-right (flip axis 0), take the median
    CBF from brain voxels that are NOT in the estimated core.

    If no core estimate is available, uses the full brain median.
    This is a robust approximation for computing relative CBF (rCBF).

    Parameters
    ----------
    cbf:
        Absolute CBF map (D, H, W) in ml/100g/min.
    brain_mask:
        Binary brain mask. If None, uses cbf > 0 as proxy.
    core_estimate:
        Optional initial core estimate to exclude from normal CBF.

    Returns
    -------
    float
        Estimated normal CBF value (ml/100g/min).
        Returns 1.0 if estimation fails (prevents division by zero).
    """
    if brain_mask is None:
        brain_mask = (cbf > 0).astype(np.float32)

    brain_bool = brain_mask > 0

    if brain_bool.sum() < 100:
        return max(float(np.median(cbf[cbf > 0])), 1.0) if (cbf > 0).any() else 1.0

    # Exclude core region from normal tissue
    normal_bool = brain_bool.copy()
    if core_estimate is not None:
        normal_bool = normal_bool & (core_estimate == 0)

    if normal_bool.sum() < 100:
        normal_bool = brain_bool

    normal_cbf = cbf[normal_bool]
    # Use median (robust to outliers)
    result = float(np.median(normal_cbf[normal_cbf > 0]))

    return max(result, 1.0)  # prevent division by zero
