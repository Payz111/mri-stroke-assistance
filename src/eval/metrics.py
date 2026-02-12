"""Segmentation evaluation metrics.

Provides voxel-level, surface-distance, and lesion-wise metrics for
comparing predicted masks against ground truth.
"""
from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute the Dice similarity coefficient.

    Parameters
    ----------
    pred:
        Binary predicted mask.
    gt:
        Binary ground-truth mask.

    Returns
    -------
    float
        Dice score in [0, 1].
    """
    raise NotImplementedError()


def hausdorff_95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute the 95th-percentile Hausdorff distance (mm).

    Parameters
    ----------
    pred:
        Binary predicted mask.
    gt:
        Binary ground-truth mask.
    spacing:
        Voxel dimensions in mm.

    Returns
    -------
    float
        HD95 distance in mm.
    """
    raise NotImplementedError()


def volume_mae(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute the mean absolute error in lesion volume (mL).

    Parameters
    ----------
    pred:
        Binary predicted mask.
    gt:
        Binary ground-truth mask.
    spacing:
        Voxel dimensions in mm.

    Returns
    -------
    float
        Absolute volume difference in mL.
    """
    raise NotImplementedError()


def lesion_wise_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.1,
) -> dict[str, float]:
    """Compute lesion-wise detection metrics (F1, precision, recall).

    Each connected component in *gt* and *pred* is treated as an
    individual lesion.  A predicted lesion is a true positive if its IoU
    with any ground-truth lesion exceeds *iou_threshold*.

    Parameters
    ----------
    pred:
        Binary predicted mask.
    gt:
        Binary ground-truth mask.
    iou_threshold:
        Minimum IoU for a match.

    Returns
    -------
    dict[str, float]
        ``{"f1", "precision", "recall", "tp", "fp", "fn"}``.
    """
    raise NotImplementedError()
