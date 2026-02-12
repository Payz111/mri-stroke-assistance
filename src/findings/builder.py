"""Findings builder — assemble the full V1 findings JSON.

Gathers outputs from volume calculation, laterality detection,
anatomic-location mapping, vascular-territory classification, and
DWI-FLAIR mismatch detection into a single structured findings
dictionary.
"""
from __future__ import annotations

from typing import Any

import numpy as np


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
    dwi:
        DWI volume array.
    adc:
        ADC volume array.
    flair:
        FLAIR volume array.
    metadata:
        Subject metadata including ``spacing``, ``affine``, ``subject_id``,
        etc.

    Returns
    -------
    dict
        Structured findings dictionary conforming to the V1 schema,
        ready for JSON serialisation and report generation.
    """
    raise NotImplementedError()
