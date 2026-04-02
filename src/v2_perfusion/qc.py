"""CTP perfusion quality control checks."""
from __future__ import annotations

import numpy as np


def check_perfusion_quality(
    maps: dict[str, np.ndarray],
    require_maps: list[str] | None = None,
    max_nan_fraction: float = 0.01,
    tmax_max_reasonable: float = 100.0,
) -> dict:
    """Check CTP perfusion maps for quality issues.

    Parameters
    ----------
    maps:
        Dictionary of perfusion maps {name: array}.
        Expected keys: "tmax", "cbf", optionally "cbv", "mtt".
    require_maps:
        List of required map names (default: ["tmax", "cbf"]).
    max_nan_fraction:
        Maximum allowed fraction of NaN voxels (default 1%).
    tmax_max_reasonable:
        Maximum reasonable Tmax value in seconds (default 100).

    Returns
    -------
    dict
        Quality gate result: {"passed": bool, "reasons": list[str]}.
    """
    if require_maps is None:
        require_maps = ["tmax", "cbf"]

    reasons = []

    # Check required maps present
    for name in require_maps:
        if name not in maps or maps[name] is None:
            reasons.append(f"Required map '{name}' is missing")

    if reasons:
        return {"passed": False, "reasons": reasons}

    # Check NaN fraction
    for name, arr in maps.items():
        if arr is None:
            continue
        nan_frac = np.isnan(arr).mean()
        if nan_frac > max_nan_fraction:
            reasons.append(
                f"Map '{name}' has {nan_frac:.1%} NaN voxels "
                f"(max {max_nan_fraction:.1%})"
            )

    # Check Tmax range
    if "tmax" in maps and maps["tmax"] is not None:
        tmax = maps["tmax"]
        tmax_valid = tmax[~np.isnan(tmax)]
        if len(tmax_valid) > 0:
            tmax_max = float(tmax_valid.max())
            if tmax_max > tmax_max_reasonable:
                reasons.append(
                    f"Tmax max={tmax_max:.1f}s exceeds "
                    f"reasonable limit ({tmax_max_reasonable}s)"
                )
            if tmax_max < 0.1:
                reasons.append("Tmax values are all near zero")

    # Check CBF has positive values in brain
    if "cbf" in maps and maps["cbf"] is not None:
        cbf = maps["cbf"]
        cbf_valid = cbf[~np.isnan(cbf)]
        positive_frac = (cbf_valid > 0).mean() if len(cbf_valid) > 0 else 0
        if positive_frac < 0.1:
            reasons.append(
                f"CBF has only {positive_frac:.1%} positive voxels"
            )

    return {"passed": len(reasons) == 0, "reasons": reasons}
