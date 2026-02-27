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
