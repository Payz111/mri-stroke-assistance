"""Error-analysis utilities for segmentation predictions.

Categorises false-positive and false-negative voxels/lesions by type
to guide targeted model improvements.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def categorize_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify false-positive and false-negative errors.

    Parameters
    ----------
    pred:
        Binary predicted mask.
    gt:
        Binary ground-truth mask.
    metadata:
        Optional subject metadata (spacing, atlas info) used to enrich
        error categorisation.

    Returns
    -------
    dict
        Error summary with keys such as ``fp_count``, ``fn_count``,
        ``fp_categories``, ``fn_categories``, and per-lesion details.
    """
    raise NotImplementedError()
