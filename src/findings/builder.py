"""Findings builder -- assemble the full V1 findings JSON.

Gathers outputs from volume calculation, laterality detection,
anatomic-location mapping, vascular-territory classification, and
DWI-FLAIR mismatch detection into a single structured findings
dictionary. Optionally validates with the Pydantic schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.ndimage import label

from src.findings.laterality import detect_laterality
from src.findings.location import map_anatomic_location
from src.findings.mismatch import detect_dwi_flair_mismatch
from src.findings.territory import classify_vascular_territory
from src.findings.volume_calc import compute_max_diameter_mm, compute_volume_ml

MODEL_VERSION = "v2.0-attention-unet3d-isles22-soop"

# Valid enum values (matching schema.py)
_VALID_LATERALITIES = {"left", "right", "bilateral"}
_VALID_TERRITORIES = {"MCA", "ACA", "PCA", "vertebrobasilar", "watershed", "uncertain"}
_VALID_LOCATIONS = {
    "frontal",
    "parietal",
    "temporal",
    "occipital",
    "insula",
    "basal_ganglia",
    "thalamus",
    "internal_capsule",
    "brainstem",
    "cerebellum",
    "multiple",
}
_VALID_FLAIR = {"none", "subtle", "definite"}
_VALID_MISMATCH = {"yes", "no", "indeterminate"}
_VALID_IMPRESSIONS = {"no_acute_infarct", "acute", "early_subacute", "indeterminate"}


def _split_lesions(combined_mask: np.ndarray) -> list[np.ndarray]:
    """Split a combined binary mask into per-lesion masks."""
    labelled, n = label(combined_mask > 0)
    masks = []
    for i in range(1, n + 1):
        masks.append((labelled == i).astype(np.float32))
    return masks


def _clamp_enum(value: str, valid: set[str], default: str) -> str:
    """Ensure value is in the valid set."""
    return value if value in valid else default


def _classify_flair_signal(flair: np.ndarray, lesion_mask: np.ndarray) -> tuple[str, float]:
    """Determine FLAIR hyperintensity within the lesion."""
    if lesion_mask.sum() == 0:
        return ("none", 0.0)

    lesion_bool = lesion_mask > 0
    brain_bool = flair > 0
    normal_bool = brain_bool & ~lesion_bool

    if normal_bool.sum() < 100:
        return ("none", 0.3)

    normal_mean = flair[normal_bool].mean()
    normal_std = max(flair[normal_bool].std(), 1e-7)
    lesion_z = (flair[lesion_bool].mean() - normal_mean) / normal_std

    if lesion_z > 2.0:
        return ("definite", min(lesion_z / 4.0, 1.0))
    if lesion_z > 1.0:
        return ("subtle", min(lesion_z / 4.0, 1.0))
    return ("none", max(1.0 - lesion_z / 2.0, 0.0))


def _check_adc_low(adc: np.ndarray, lesion_mask: np.ndarray) -> tuple[bool, float]:
    """Check if ADC is reduced within the lesion (acute ischemia sign)."""
    if lesion_mask.sum() == 0:
        return (False, 0.0)

    lesion_bool = lesion_mask > 0
    brain_bool = adc > 0
    normal_bool = brain_bool & ~lesion_bool

    if normal_bool.sum() < 100:
        return (False, 0.3)

    normal_mean = adc[normal_bool].mean()
    normal_std = max(adc[normal_bool].std(), 1e-7)
    lesion_mean = adc[lesion_bool].mean()

    z_below = (normal_mean - lesion_mean) / normal_std

    if z_below > 1.5:
        return (True, min(z_below / 3.0, 1.0))
    return (False, max(0.0, z_below / 3.0))


def _round(value: Any, digits: int) -> float:
    """Round *value* and coerce numpy scalars to a JSON-serialisable float.

    The extractors return numpy float32 scalars, which ``json.dumps`` cannot
    encode -- keep the findings dict pure-Python so it serialises as numbers.
    """
    return round(float(value), digits)


def _find_evidence_slices(mask: np.ndarray, top_k: int = 3) -> list[int]:
    """Find axial slices with the most lesion voxels."""
    if mask.ndim != 3 or mask.sum() == 0:
        return []

    per_slice = (mask > 0).sum(axis=(0, 1))
    indices = np.argsort(-per_slice)[:top_k]
    return sorted(int(i) for i in indices if per_slice[i] > 0)


def _map_mismatch_status(status: str) -> str:
    """Convert mismatch detector output to schema enum value."""
    return {"positive": "yes", "negative": "no"}.get(status, "indeterminate")


def _determine_impression(
    lesions: list[dict],
) -> tuple[str, float]:
    """Determine overall impression from lesion findings."""
    if not lesions:
        return ("no_acute_infarct", 0.8)

    has_adc_low = any(les.get("adc_low_confirmed") for les in lesions)
    has_mismatch_yes = any(les.get("dwi_flair_mismatch") == "yes" for les in lesions)
    has_mismatch_no = any(les.get("dwi_flair_mismatch") == "no" for les in lesions)

    if has_adc_low and has_mismatch_yes:
        return ("acute", 0.80)
    if has_adc_low and has_mismatch_no:
        return ("early_subacute", 0.65)
    if has_adc_low:
        return ("acute", 0.60)
    return ("indeterminate", 0.40)


def build_findings(
    lesion_masks: list[np.ndarray],
    dwi: np.ndarray,
    adc: np.ndarray,
    flair: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the complete V1Findings JSON-compatible dictionary.

    Parameters
    ----------
    lesion_masks:
        List of per-lesion binary masks (one per detected lesion).
        If a single combined mask is provided (list of one), it will
        be automatically split into connected components.
    dwi:
        DWI volume array (3-D).
    adc:
        ADC volume array (3-D).
    flair:
        FLAIR volume array (3-D).
    metadata:
        Subject metadata including ``spacing``, ``affine``, ``subject_id``.

    Returns
    -------
    dict
        Structured findings dictionary conforming to the V1 schema,
        ready for JSON serialisation and report generation.
    """
    spacing = tuple(float(s) for s in metadata.get("spacing", (1.0, 1.0, 1.0)))
    subject_id = metadata.get("subject_id", "unknown")

    # Auto-split combined mask if only one is provided
    if len(lesion_masks) == 1:
        lesion_masks = _split_lesions(lesion_masks[0])

    # Build per-lesion findings
    lesions = []
    total_volume = 0.0

    for idx, lmask in enumerate(lesion_masks):
        vol_ml = compute_volume_ml(lmask, spacing)
        max_diam = compute_max_diameter_mm(lmask, spacing)
        lat = detect_laterality(lmask)
        locations = map_anatomic_location(lmask)
        territory, terr_conf = classify_vascular_territory(lmask)
        mismatch_status, mismatch_score = detect_dwi_flair_mismatch(dwi, flair, lmask)
        flair_signal, flair_conf = _classify_flair_signal(flair, lmask)
        adc_low, adc_conf = _check_adc_low(adc, lmask)
        evidence = _find_evidence_slices(lmask)

        lesion = {
            "lesion_id": idx + 1,
            "mask_ref": f"lesion_{idx + 1}_mask.nii.gz",
            "volume_ml": _round(vol_ml, 2),
            "max_diameter_mm": _round(max_diam, 1),
            "laterality": _clamp_enum(lat, _VALID_LATERALITIES, "bilateral"),
            "anatomic_location": [
                _clamp_enum(loc, _VALID_LOCATIONS, "multiple")
                for loc in (locations or ["multiple"])
            ],
            "vascular_territory": _clamp_enum(territory, _VALID_TERRITORIES, "uncertain"),
            "territory_confidence": _round(terr_conf, 2),
            "adc_low_confirmed": bool(adc_low),
            "adc_confidence": _round(adc_conf, 2),
            "flair_hyperintensity": _clamp_enum(flair_signal, _VALID_FLAIR, "none"),
            "flair_confidence": _round(flair_conf, 2),
            "dwi_flair_mismatch": _clamp_enum(
                _map_mismatch_status(mismatch_status),
                _VALID_MISMATCH,
                "indeterminate",
            ),
            "mismatch_confidence": _round(mismatch_score, 2),
            "evidence_slices": {
                "series_uid": None,
                "slice_indices": evidence,
            },
        }
        lesions.append(lesion)
        total_volume += vol_ml

    impression, impression_conf = _determine_impression(lesions)

    findings = {
        "study_id": subject_id,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol_present": {
            "dwi": True,
            "adc": True,
            "flair": True,
            "gre_swi": False,
            "tof_mra": False,
        },
        "quality_gate": {"passed": True, "reasons": []},
        "lesions": lesions,
        "total_lesion_count": len(lesions),
        "total_lesion_volume_ml": _round(total_volume, 2),
        "overall_impression": _clamp_enum(impression, _VALID_IMPRESSIONS, "indeterminate"),
        "overall_confidence": _round(impression_conf, 2),
        "combined_mask_ref": f"{subject_id}_combined_mask.nii.gz",
    }

    return findings
