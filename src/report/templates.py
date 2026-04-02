"""Report template strings.

Contains the text templates used by the report generator to convert
structured findings into human-readable radiology-style prose.

Key principle: ZERO hallucinations -- every sentence is directly
derived from structured JSON fields. No free-text generation.
"""
from __future__ import annotations

HEADER_TEMPLATE = """\
MRI BRAIN - STROKE PROTOCOL
Study ID: {study_id}
Model: {model_version}
Generated: {timestamp}
"""

PROTOCOL_TEMPLATE = """\
TECHNIQUE:
MRI brain with stroke protocol. Sequences available: {sequences}.
"""

NO_LESION_FINDINGS = """\
FINDINGS:
No acute ischemic lesion identified on DWI/ADC sequences.
"""

LESION_SUMMARY_TEMPLATE = """\
FINDINGS:
{n_lesions} ischemic lesion(s) identified, total volume {total_volume_ml} mL.
"""

LESION_DETAIL_TEMPLATE = """\
Lesion {lesion_id}: {volume_ml} mL ({max_diameter_mm} mm max diameter), \
{laterality} {location}, {territory} territory{territory_note}. \
ADC {adc_status}{flair_note}. \
DWI-FLAIR mismatch: {mismatch_status}.
"""

IMPRESSION_TEMPLATE = """\
IMPRESSION:
{impression_body}
"""

QC_FAIL_TEMPLATE = """\
*** QUALITY CHECK FAILED ***
Reasons: {reasons}
This report should be reviewed with caution.
"""

# Mapping enums to readable text
LATERALITY_TEXT = {
    "left": "left hemispheric",
    "right": "right hemispheric",
    "bilateral": "bilateral",
}

IMPRESSION_TEXT = {
    "acute": "Acute ischemic infarct",
    "early_subacute": "Early subacute ischemic infarct",
    "no_acute_infarct": "No acute ischemic infarct identified",
    "indeterminate": "Indeterminate findings, clinical correlation recommended",
}

MISMATCH_TEXT = {
    "yes": "positive (suggests <4.5h onset)",
    "no": "negative",
    "indeterminate": "indeterminate",
}

ADC_TEXT = {
    True: "restricted (confirms acute ischemia)",
    False: "not convincingly restricted",
}

FLAIR_TEXT = {
    "definite": "FLAIR hyperintensity present (definite)",
    "subtle": "FLAIR signal subtle/equivocal",
    "none": "No FLAIR hyperintensity",
}

# ---------------------------------------------------------------------------
# V2 Perfusion templates
# ---------------------------------------------------------------------------

PERFUSION_HEADER_TEMPLATE = """\
CT PERFUSION ANALYSIS:
Perfusion source: {source}. Maps available: {maps_available}.
Thresholds: Tmax >= {tmax_threshold}s (hypoperfusion), rCBF <= {rcbf_pct}% (core).
"""

PERFUSION_CORE_TEMPLATE = """\
Ischemic core (rCBF <= {rcbf_pct}%): {core_volume_ml} mL.
"""

PERFUSION_HYPOPERFUSION_TEMPLATE = """\
Hypoperfusion region (Tmax >= {tmax_threshold}s): {hypo_volume_ml} mL.
"""

PERFUSION_PENUMBRA_TEMPLATE = """\
Penumbra (tissue at risk): {penumbra_volume_ml} mL.
"""

PERFUSION_MISMATCH_TEMPLATE = """\
Mismatch ratio: {ratio} (hypoperfusion/core). {status_text}
"""

PERFUSION_QC_FAIL_TEMPLATE = """\
*** PERFUSION QUALITY CHECK FAILED ***
Reasons: {reasons}
Perfusion analysis results should be interpreted with caution.
"""

TARGET_MISMATCH_TEXT = {
    "target_mismatch": (
        "TARGET MISMATCH PRESENT -- patient may benefit from "
        "reperfusion therapy (core <{core_max}mL, ratio >{ratio_min}, "
        "mismatch >{vol_min}mL)."
    ),
    "no_mismatch": "No target mismatch identified.",
    "indeterminate": (
        "Mismatch assessment indeterminate, clinical correlation required."
    ),
}

PERFUSION_IMPRESSION_TEMPLATE = """\
CT Perfusion: core {core_vol} mL, penumbra {penumbra_vol} mL, \
mismatch ratio {ratio} ({status}).
"""
