"""CTP perfusion inference pipeline.

Threshold-based analysis (no deep learning model needed):
Tmax + CBF -> core/penumbra/mismatch -> findings -> report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.findings.v2_builder import build_v2_findings
from src.inference.ctp_visualize import create_perfusion_montage
from src.report.generator import generate_report
from src.report.validator import validate_report

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "v2_ctp.yaml"


def load_ctp_config(config_path: str | Path | None = None) -> dict:
    """Load CTP configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def run_ctp_inference(
    tmax: np.ndarray,
    cbf: np.ndarray,
    metadata: dict[str, Any],
    ncct: np.ndarray | None = None,
    cbv: np.ndarray | None = None,
    mtt: np.ndarray | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full CTP perfusion analysis pipeline.

    This is threshold-based (no model needed). Steps:
    1. Load config
    2. Build V2 perfusion findings (core/penumbra/mismatch)
    3. Generate report with perfusion section
    4. Validate report
    5. Create visualization

    Parameters
    ----------
    tmax:
        Tmax perfusion map (D, H, W) in seconds.
    cbf:
        CBF perfusion map (D, H, W).
    metadata:
        Subject metadata with ``spacing``, ``subject_id``.
    ncct:
        Optional NCCT volume for brain mask and visualization.
    cbv, mtt:
        Optional additional perfusion maps.
    config_path:
        Path to v2_ctp.yaml config file.

    Returns
    -------
    dict
        Result with keys: findings, report, validation, preview,
        core_mask, penumbra_mask, hypoperfusion_mask.
    """
    config = load_ctp_config(config_path)

    # Build V2 findings
    v2_findings = build_v2_findings(
        tmax=tmax,
        cbf=cbf,
        metadata=metadata,
        ncct=ncct,
        cbv=cbv,
        mtt=mtt,
        config=config,
    )

    # Extract masks before cleaning for report
    masks = v2_findings.pop("_masks", {})

    # Wrap in a full findings dict for report generation
    # (V2-only mode: no V1 lesion data)
    findings = {
        "study_id": metadata.get("subject_id", "unknown"),
        "model_version": "v2.0-ctp-threshold",
        "timestamp": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "protocol_present": {
            "dwi": False,
            "adc": False,
            "flair": False,
            "gre_swi": False,
            "tof_mra": False,
        },
        "quality_gate": {"passed": True, "reasons": []},
        "lesions": [],
        "total_lesion_count": 0,
        "total_lesion_volume_ml": 0.0,
        "overall_impression": "indeterminate",
        "overall_confidence": 0.5,
        "combined_mask_ref": "",
        "perfusion": v2_findings,
    }

    # Generate report
    report = generate_report(findings)

    # Validate
    validation = validate_report(report, findings)

    # Visualize
    bg = ncct if ncct is not None else tmax
    core_mask = masks.get("core", np.zeros_like(tmax))
    penumbra_mask = masks.get("penumbra", np.zeros_like(tmax))

    preview = create_perfusion_montage(
        background=bg,
        core_mask=core_mask,
        penumbra_mask=penumbra_mask,
        tmax=tmax,
        n_slices=6,
    )

    return {
        "findings": findings,
        "report": report,
        "validation": validation,
        "preview": preview,
        "core_mask": core_mask,
        "penumbra_mask": penumbra_mask,
        "hypoperfusion_mask": masks.get("hypoperfusion", np.zeros_like(tmax)),
    }
