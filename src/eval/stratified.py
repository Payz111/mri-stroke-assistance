"""Stratified evaluation across clinical sub-groups.

Breaks down segmentation performance by lesion volume so that
per-subgroup weaknesses can be identified.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.eval.metrics import compute_all_metrics, volume_ml

# Same bins as EDA (TASK-002)
VOLUME_BINS = {
    "tiny (<1ml)": (0, 1),
    "small (1-10ml)": (1, 10),
    "medium (10-50ml)": (10, 50),
    "large (>50ml)": (50, float("inf")),
}


def _classify_volume(vol_ml: float) -> str:
    """Assign a volume to a size category."""
    for name, (lo, hi) in VOLUME_BINS.items():
        if lo <= vol_ml < hi:
            return name
    return "large (>50ml)"


def stratified_evaluation(
    predictions: Sequence[np.ndarray],
    ground_truths: Sequence[np.ndarray],
    metadata: Sequence[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predictions stratified by lesion volume.

    Parameters
    ----------
    predictions:
        List of predicted binary masks (one per subject).
    ground_truths:
        Corresponding ground-truth masks.
    metadata:
        Per-subject metadata dicts (must include ``spacing``).

    Returns
    -------
    dict
        ``{"overall": {...}, "by_size": {category: {...}}, "per_subject": [...]}``
    """
    per_subject = []
    by_size: dict[str, list[dict]] = {name: [] for name in VOLUME_BINS}

    for pred, gt, meta in zip(predictions, ground_truths, metadata):
        spacing = tuple(float(s) for s in meta.get("spacing", (1.0, 1.0, 1.0)))
        subject_id = meta.get("subject_id", "unknown")

        metrics = compute_all_metrics(pred, gt, spacing)
        metrics["subject_id"] = subject_id

        gt_vol = volume_ml(gt, spacing)
        size_cat = _classify_volume(gt_vol)
        metrics["size_category"] = size_cat

        per_subject.append(metrics)
        by_size[size_cat].append(metrics)

    # Aggregate overall
    metric_keys = ["dice", "iou", "sensitivity", "hd95", "volume_mae_ml", "lesion_f1"]
    overall = {}
    for key in metric_keys:
        values = [s[key] for s in per_subject if np.isfinite(s[key])]
        if values:
            overall[key + "_mean"] = float(np.mean(values))
            overall[key + "_std"] = float(np.std(values))
            overall[key + "_median"] = float(np.median(values))

    overall["n_subjects"] = len(per_subject)

    # Aggregate by size
    by_size_summary = {}
    for cat_name, cat_subjects in by_size.items():
        if not cat_subjects:
            by_size_summary[cat_name] = {"n": 0}
            continue

        cat_summary = {"n": len(cat_subjects)}
        for key in metric_keys:
            values = [s[key] for s in cat_subjects if np.isfinite(s[key])]
            if values:
                cat_summary[key + "_mean"] = float(np.mean(values))
                cat_summary[key + "_std"] = float(np.std(values))
        by_size_summary[cat_name] = cat_summary

    return {
        "overall": overall,
        "by_size": by_size_summary,
        "per_subject": per_subject,
    }
