"""Threshold-based perfusion map analysis.

Clinical standard approach for CTP:
- Hypoperfusion: Tmax >= 6 seconds
- Ischemic core: rCBF <= 30%
- Penumbra (tissue at risk): hypoperfusion minus core
- Target mismatch: criteria for reperfusion therapy eligibility
"""
from __future__ import annotations

import numpy as np

from src.findings.volume_calc import compute_volume_ml
from src.v2_perfusion.contralateral import estimate_contralateral_cbf


def compute_hypoperfusion_mask(
    tmax: np.ndarray,
    threshold_sec: float = 6.0,
    brain_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute hypoperfusion mask from Tmax map.

    Hypoperfusion is defined as Tmax >= threshold (default 6 seconds).

    Parameters
    ----------
    tmax:
        Tmax perfusion map (D, H, W) in seconds.
    threshold_sec:
        Tmax threshold in seconds (default 6.0).
    brain_mask:
        Binary brain mask. If None, uses tmax > 0 as proxy.

    Returns
    -------
    np.ndarray
        Binary hypoperfusion mask (D, H, W) float32.
    """
    mask = (tmax >= threshold_sec).astype(np.float32)

    if brain_mask is not None:
        mask = mask * (brain_mask > 0).astype(np.float32)
    else:
        # Exclude background (tmax == 0 is likely outside brain)
        mask = mask * (tmax > 0).astype(np.float32)

    return mask


def compute_core_mask(
    cbf: np.ndarray,
    threshold_rcbf: float = 0.30,
    brain_mask: np.ndarray | None = None,
    normal_cbf: float | None = None,
) -> np.ndarray:
    """Compute ischemic core mask from CBF map.

    Core is defined as rCBF <= threshold (default 30%).
    rCBF = CBF / normal_CBF (contralateral hemisphere).

    Parameters
    ----------
    cbf:
        CBF perfusion map (D, H, W). Can be absolute or relative.
    threshold_rcbf:
        Relative CBF threshold (default 0.30 = 30%).
    brain_mask:
        Binary brain mask. If None, uses cbf > 0 as proxy.
    normal_cbf:
        Normal CBF value for rCBF computation.
        If None, estimated from contralateral hemisphere.

    Returns
    -------
    np.ndarray
        Binary core mask (D, H, W) float32.
    """
    if normal_cbf is None:
        normal_cbf = estimate_contralateral_cbf(cbf, brain_mask)

    # Compute relative CBF
    rcbf = np.where(normal_cbf > 0, cbf / normal_cbf, 0.0)

    # Core = rCBF <= threshold AND within brain
    mask = (rcbf <= threshold_rcbf).astype(np.float32)

    if brain_mask is not None:
        mask = mask * (brain_mask > 0).astype(np.float32)
    else:
        mask = mask * (cbf > 0).astype(np.float32)

    return mask


def compute_penumbra_mask(
    hypoperfusion_mask: np.ndarray,
    core_mask: np.ndarray,
) -> np.ndarray:
    """Compute penumbra mask (tissue at risk).

    Penumbra = hypoperfusion region minus ischemic core.
    This is the potentially salvageable tissue.

    Parameters
    ----------
    hypoperfusion_mask:
        Binary hypoperfusion mask (Tmax >= 6s).
    core_mask:
        Binary core mask (rCBF <= 30%).

    Returns
    -------
    np.ndarray
        Binary penumbra mask (D, H, W) float32.
    """
    return ((hypoperfusion_mask > 0) & (core_mask == 0)).astype(np.float32)


def compute_mismatch_metrics(
    core_mask: np.ndarray,
    hypoperfusion_mask: np.ndarray,
    spacing: tuple[float, float, float],
    core_max_ml: float = 70.0,
    mismatch_ratio_min: float = 1.8,
    mismatch_volume_min_ml: float = 15.0,
) -> dict:
    """Compute core-to-hypoperfusion mismatch metrics.

    Evaluates target mismatch criteria for reperfusion therapy:
    1. Core volume < 70 mL
    2. Mismatch ratio > 1.8
    3. Mismatch volume > 15 mL

    All three must be met for "target_mismatch" status.

    Parameters
    ----------
    core_mask:
        Binary core mask.
    hypoperfusion_mask:
        Binary hypoperfusion mask.
    spacing:
        Voxel dimensions in mm (sx, sy, sz).
    core_max_ml:
        Maximum core volume for mismatch (default 70 mL).
    mismatch_ratio_min:
        Minimum mismatch ratio (default 1.8).
    mismatch_volume_min_ml:
        Minimum mismatch volume (default 15 mL).

    Returns
    -------
    dict
        Mismatch metrics matching MismatchMetrics schema.
    """
    core_vol = compute_volume_ml(core_mask, spacing)
    hypo_vol = compute_volume_ml(hypoperfusion_mask, spacing)
    mismatch_vol = max(hypo_vol - core_vol, 0.0)

    # Mismatch ratio (avoid division by zero)
    if core_vol > 0.1:
        ratio = hypo_vol / core_vol
    elif hypo_vol > 0.1:
        ratio = float("inf")
    else:
        ratio = 1.0

    # Evaluate target mismatch criteria
    criteria = {
        "core_below_max": core_vol < core_max_ml,
        "ratio_above_min": ratio > mismatch_ratio_min,
        "volume_above_min": mismatch_vol > mismatch_volume_min_ml,
    }

    all_met = all(criteria.values())
    if core_vol < 0.1 and hypo_vol < 0.1:
        status = "indeterminate"
        confidence = 0.3
    elif all_met:
        status = "target_mismatch"
        confidence = 0.85
    else:
        status = "no_mismatch"
        confidence = 0.75

    return {
        "mismatch_volume_ml": round(mismatch_vol, 2),
        "mismatch_ratio": round(min(ratio, 99.9), 1),
        "target_mismatch_status": status,
        "criteria_used": {
            "core_max_ml": core_max_ml,
            "mismatch_ratio_min": mismatch_ratio_min,
            "mismatch_volume_min_ml": mismatch_volume_min_ml,
        },
        "core_volume_ml": round(core_vol, 2),
        "hypoperfusion_volume_ml": round(hypo_vol, 2),
        "confidence": confidence,
    }
