"""V2 Perfusion findings builder.

Assembles V2PerfusionFindings-compatible dict from CTP perfusion maps.
Uses threshold-based approach (clinical standard):
- Hypoperfusion: Tmax >= 6s
- Core: rCBF <= 30%
- Target mismatch assessment

Follows the same plain-dict pattern as builder.py (no Pydantic runtime).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.findings.volume_calc import compute_volume_ml
from src.v2_perfusion.brain_mask import estimate_brain_mask_ct
from src.v2_perfusion.contralateral import estimate_contralateral_cbf
from src.v2_perfusion.qc import check_perfusion_quality
from src.v2_perfusion.threshold_maps import (
    compute_core_mask,
    compute_hypoperfusion_mask,
    compute_mismatch_metrics,
    compute_penumbra_mask,
)

# Default thresholds (matching configs/v2_ctp.yaml)
DEFAULT_TMAX_THRESHOLD = 6.0
DEFAULT_RCBF_THRESHOLD = 0.30
DEFAULT_CORE_MAX_ML = 70.0
DEFAULT_MISMATCH_RATIO_MIN = 1.8
DEFAULT_MISMATCH_VOLUME_MIN_ML = 15.0


def build_v2_findings(
    tmax: np.ndarray,
    cbf: np.ndarray,
    metadata: dict[str, Any],
    ncct: np.ndarray | None = None,
    cbv: np.ndarray | None = None,
    mtt: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build V2 perfusion findings dict from CTP maps.

    Parameters
    ----------
    tmax:
        Tmax perfusion map (D, H, W) in seconds.
    cbf:
        CBF perfusion map (D, H, W) in ml/100g/min.
    metadata:
        Subject metadata including ``spacing``, ``subject_id``.
    ncct:
        Optional NCCT volume for brain mask estimation.
    cbv:
        Optional CBV perfusion map.
    mtt:
        Optional MTT perfusion map.
    config:
        Optional config dict with thresholds. If None, uses defaults.

    Returns
    -------
    dict
        V2PerfusionFindings-compatible dictionary.
    """
    spacing = tuple(float(s) for s in metadata.get("spacing", (1.0, 1.0, 1.0)))

    # Parse config
    if config is None:
        config = {}
    thresholds = config.get("thresholds", {})
    hypo_cfg = thresholds.get("hypoperfusion", {})
    core_cfg = thresholds.get("core", {})
    mismatch_cfg = config.get("mismatch", {}).get("target_profile", {})
    qc_cfg = config.get("perfusion_qc", {})

    tmax_threshold = hypo_cfg.get("value_seconds", DEFAULT_TMAX_THRESHOLD)
    rcbf_threshold = core_cfg.get("value_rcbf", DEFAULT_RCBF_THRESHOLD)

    # Step 1: Quality check
    maps = {"tmax": tmax, "cbf": cbf}
    if cbv is not None:
        maps["cbv"] = cbv
    if mtt is not None:
        maps["mtt"] = mtt

    qc_result = check_perfusion_quality(
        maps,
        require_maps=qc_cfg.get("require_maps", ["tmax", "cbf"]),
        max_nan_fraction=qc_cfg.get("max_nan_fraction", 0.01),
        tmax_max_reasonable=qc_cfg.get("tmax_max_reasonable", 100.0),
    )

    # Step 2: Brain mask
    if ncct is not None:
        brain_mask = estimate_brain_mask_ct(ncct)
    else:
        brain_mask = (tmax > 0).astype(np.float32)

    # Step 3: Hypoperfusion mask
    hypo_mask = compute_hypoperfusion_mask(tmax, tmax_threshold, brain_mask)

    # Step 4: Contralateral CBF + core mask
    normal_cbf = estimate_contralateral_cbf(cbf, brain_mask)
    core_mask = compute_core_mask(cbf, rcbf_threshold, brain_mask, normal_cbf)

    # Step 5: Penumbra
    penumbra_mask = compute_penumbra_mask(hypo_mask, core_mask)

    # Step 6: Mismatch metrics
    mismatch = compute_mismatch_metrics(
        core_mask,
        hypo_mask,
        spacing,
        core_max_ml=mismatch_cfg.get("core_max_ml", DEFAULT_CORE_MAX_ML),
        mismatch_ratio_min=mismatch_cfg.get(
            "mismatch_ratio_min", DEFAULT_MISMATCH_RATIO_MIN
        ),
        mismatch_volume_min_ml=mismatch_cfg.get(
            "mismatch_volume_min_ml", DEFAULT_MISMATCH_VOLUME_MIN_ML
        ),
    )

    # Volumes
    core_vol = compute_volume_ml(core_mask, spacing)
    hypo_vol = compute_volume_ml(hypo_mask, spacing)
    penumbra_vol = compute_volume_ml(penumbra_mask, spacing)

    # Assemble findings dict
    subject_id = metadata.get("subject_id", "unknown")

    findings = {
        "perfusion_source": "CTP",
        "maps_present": {
            "tmax": True,
            "cbf": True,
            "cbv": cbv is not None,
            "mtt": mtt is not None,
        },
        "thresholds_used": {
            "hypoperfusion_tmax_sec": tmax_threshold,
            "core_rcbf_threshold": rcbf_threshold,
        },
        "perfusion_quality_gate": qc_result,
        "core": {
            "volume_ml": round(core_vol, 2),
            "mask_ref": f"{subject_id}_core_mask.nii.gz",
            "method": "rCBF_threshold",
            "confidence": 0.80 if qc_result["passed"] else 0.40,
        },
        "hypoperfusion": {
            "volume_ml": round(hypo_vol, 2),
            "mask_ref": f"{subject_id}_hypoperfusion_mask.nii.gz",
            "method": "Tmax_threshold",
            "confidence": 0.80 if qc_result["passed"] else 0.40,
        },
        "penumbra": {
            "volume_ml": round(penumbra_vol, 2),
            "mask_ref": f"{subject_id}_penumbra_mask.nii.gz",
        },
        "mismatch": mismatch,
        # Return masks for visualization
        "_masks": {
            "core": core_mask,
            "hypoperfusion": hypo_mask,
            "penumbra": penumbra_mask,
            "brain": brain_mask,
        },
    }

    return findings


def build_combined_findings(
    v1_findings: dict[str, Any],
    v2_findings: dict[str, Any],
) -> dict[str, Any]:
    """Merge V1 structural findings with V2 perfusion findings.

    Parameters
    ----------
    v1_findings:
        V1 findings dict from builder.build_findings().
    v2_findings:
        V2 findings dict from build_v2_findings().

    Returns
    -------
    dict
        Combined findings with all V1 fields plus a "perfusion" key.
    """
    combined = dict(v1_findings)
    # Remove internal masks from v2 before merging
    v2_clean = {k: v for k, v in v2_findings.items() if not k.startswith("_")}
    combined["perfusion"] = v2_clean
    return combined
