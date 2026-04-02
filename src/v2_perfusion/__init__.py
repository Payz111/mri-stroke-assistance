"""V2: CT Perfusion analysis module.

Threshold-based core/penumbra/mismatch analysis for CT Perfusion data.
Clinical standard approach (not deep learning):
- Hypoperfusion: Tmax >= 6 seconds
- Ischemic core: rCBF <= 30%
- Penumbra: hypoperfusion minus core
- Target mismatch: criteria for reperfusion therapy eligibility
"""
from src.v2_perfusion.brain_mask import estimate_brain_mask_ct
from src.v2_perfusion.contralateral import estimate_contralateral_cbf
from src.v2_perfusion.qc import check_perfusion_quality
from src.v2_perfusion.threshold_maps import (
    compute_core_mask,
    compute_hypoperfusion_mask,
    compute_mismatch_metrics,
    compute_penumbra_mask,
)

__all__ = [
    "estimate_brain_mask_ct",
    "estimate_contralateral_cbf",
    "check_perfusion_quality",
    "compute_hypoperfusion_mask",
    "compute_core_mask",
    "compute_penumbra_mask",
    "compute_mismatch_metrics",
]
