"""Error-analysis utilities for segmentation predictions.

Categorises false-positive and false-negative voxels/lesions by type
to guide targeted model improvements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label

from src.eval.metrics import volume_ml


def categorize_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify false-positive and false-negative errors.

    Returns
    -------
    dict
        Error summary with fp/fn counts and per-lesion details.
    """
    spacing = (1.0, 1.0, 1.0)
    if metadata and "spacing" in metadata:
        spacing = tuple(float(s) for s in metadata["spacing"])

    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)

    fp_mask = pred_bin & ~gt_bin
    fn_mask = ~pred_bin & gt_bin

    fp_voxels = int(fp_mask.sum())
    fn_voxels = int(fn_mask.sum())

    # Lesion-level analysis
    gt_labels, n_gt = label(gt_bin)
    pred_labels, n_pred = label(pred_bin)

    # Missed lesions (FN lesions): GT lesions with no overlap with prediction
    missed_lesions = []
    for i in range(1, n_gt + 1):
        lesion_mask = gt_labels == i
        overlap = (lesion_mask & pred_bin).sum()
        if overlap == 0:
            missed_lesions.append({
                "gt_lesion_id": i,
                "volume_ml": volume_ml(lesion_mask.astype(np.float32), spacing),
                "voxels": int(lesion_mask.sum()),
            })

    # Spurious lesions (FP lesions): pred lesions with no overlap with GT
    spurious_lesions = []
    for i in range(1, n_pred + 1):
        lesion_mask = pred_labels == i
        overlap = (lesion_mask & gt_bin).sum()
        if overlap == 0:
            spurious_lesions.append({
                "pred_lesion_id": i,
                "volume_ml": volume_ml(lesion_mask.astype(np.float32), spacing),
                "voxels": int(lesion_mask.sum()),
            })

    return {
        "fp_voxels": fp_voxels,
        "fn_voxels": fn_voxels,
        "fp_volume_ml": volume_ml(fp_mask.astype(np.float32), spacing),
        "fn_volume_ml": volume_ml(fn_mask.astype(np.float32), spacing),
        "n_gt_lesions": n_gt,
        "n_pred_lesions": n_pred,
        "n_missed_lesions": len(missed_lesions),
        "n_spurious_lesions": len(spurious_lesions),
        "missed_lesions": missed_lesions,
        "spurious_lesions": spurious_lesions,
    }
