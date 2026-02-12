"""Stratified evaluation across clinical sub-groups.

Breaks down segmentation performance by lesion volume, location,
territory, and other metadata dimensions so that per-subgroup
weaknesses can be identified.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def stratified_evaluation(
    predictions: Sequence[np.ndarray],
    ground_truths: Sequence[np.ndarray],
    metadata: Sequence[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predictions stratified by clinical sub-groups.

    Parameters
    ----------
    predictions:
        List of predicted binary masks (one per subject).
    ground_truths:
        Corresponding ground-truth masks.
    metadata:
        Per-subject metadata dicts (must include ``spacing`` and any
        stratification keys requested via *config*).
    config:
        Specifies stratification axes (e.g. ``volume_bins``,
        ``territory``, ``laterality``).

    Returns
    -------
    dict
        Nested dictionary ``{stratum_name: {metric: value, ...}, ...}``
        with overall and per-stratum metric summaries.
    """
    raise NotImplementedError()
