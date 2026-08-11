"""Segmentation evaluation metrics.

Provides voxel-level, surface-distance, and lesion-wise metrics for
comparing predicted masks against ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label


def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-7) -> float:
    """Compute the Dice similarity coefficient.

    Returns 1.0 if both masks are empty.
    """
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    intersection = (pred * gt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + gt.sum() + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-7) -> float:
    """Compute Intersection over Union (Jaccard index)."""
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute sensitivity (recall / true positive rate)."""
    gt_pos = (gt > 0).sum()
    if gt_pos == 0:
        return 1.0
    tp = ((pred > 0) & (gt > 0)).sum()
    return float(tp / gt_pos)


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    gt_neg = (gt == 0).sum()
    if gt_neg == 0:
        return 1.0
    tn = ((pred == 0) & (gt == 0)).sum()
    return float(tn / gt_neg)


def hausdorff_95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute the 95th-percentile Hausdorff distance (mm).

    Returns inf if one mask is empty and the other is not.
    Returns 0.0 if both masks are empty.
    """
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")

    # Distance from pred surface to gt surface and vice versa
    pred_border = pred ^ _erode(pred)
    gt_border = gt ^ _erode(gt)

    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    dist_pred_to_gt = dt_gt[pred_border]
    dist_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple binary erosion by 1 voxel (6-connected)."""
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, iterations=1)


def volume_ml(
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute lesion volume in mL from a binary mask."""
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    return float((mask > 0).sum() * voxel_vol_mm3 / 1000.0)


def volume_mae(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute the absolute error in lesion volume (mL)."""
    pred_vol = volume_ml(pred, spacing)
    gt_vol = volume_ml(gt, spacing)
    return abs(pred_vol - gt_vol)


def lesion_wise_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.1,
) -> dict[str, float]:
    """Compute lesion-wise detection metrics (F1, precision, recall).

    Each connected component is treated as an individual lesion.
    A predicted lesion is a true positive if its IoU with any
    ground-truth lesion exceeds iou_threshold.
    """
    pred_labels, n_pred = label(pred > 0)
    gt_labels, n_gt = label(gt > 0)

    if n_gt == 0 and n_pred == 0:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_gt == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 1.0, "tp": 0, "fp": n_pred, "fn": 0}
    if n_pred == 0:
        return {"f1": 0.0, "precision": 1.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": n_gt}

    # Check each predicted lesion against GT lesions
    tp = 0
    matched_gt = set()
    for i in range(1, n_pred + 1):
        pred_mask = pred_labels == i
        best_iou = 0.0
        best_gt = -1
        for j in range(1, n_gt + 1):
            gt_mask = gt_labels == j
            inter = (pred_mask & gt_mask).sum()
            union = pred_mask.sum() + gt_mask.sum() - inter
            if union > 0:
                cur_iou = inter / union
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt = j
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)

    fp = n_pred - tp
    fn = n_gt - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """Compute all metrics for a single subject."""
    result = {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "sensitivity": sensitivity(pred, gt),
        "specificity": specificity(pred, gt),
        "hd95": hausdorff_95(pred, gt, spacing),
        "volume_mae_ml": volume_mae(pred, gt, spacing),
        "pred_volume_ml": volume_ml(pred, spacing),
        "gt_volume_ml": volume_ml(gt, spacing),
    }
    lesion_metrics = lesion_wise_metrics(pred, gt)
    result["lesion_f1"] = lesion_metrics["f1"]
    result["lesion_precision"] = lesion_metrics["precision"]
    result["lesion_recall"] = lesion_metrics["recall"]
    return result
