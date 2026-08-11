"""Tests for segmentation metrics.

Every published number in docs/EVALUATION_REPORT.md comes out of this module,
so the cases below check against values worked out by hand rather than against
whatever the implementation happens to return.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    compute_all_metrics,
    dice_score,
    hausdorff_95,
    iou_score,
    lesion_wise_metrics,
    sensitivity,
    specificity,
    volume_mae,
    volume_ml,
)

SHAPE = (20, 20, 20)


def block(z=slice(5, 10), y=slice(5, 10), x=slice(5, 10)) -> np.ndarray:
    """A cuboid mask; the default is 5x5x5 = 125 voxels."""
    m = np.zeros(SHAPE, dtype=np.float32)
    m[z, y, x] = 1.0
    return m


class TestDice:
    def test_identical_masks(self):
        m = block()
        assert dice_score(m, m) == pytest.approx(1.0, abs=1e-6)

    def test_both_empty_is_one(self):
        """No lesion predicted on a study with no lesion is a perfect result."""
        empty = np.zeros(SHAPE, dtype=np.float32)
        assert dice_score(empty, empty) == 1.0

    def test_disjoint_masks(self):
        pred = block(z=slice(0, 5))
        gt = block(z=slice(10, 15))
        assert dice_score(pred, gt) == pytest.approx(0.0, abs=1e-6)

    def test_half_overlap(self):
        """|A|=|B|=100, |A and B|=50 -> 2*50/200 = 0.5."""
        pred = np.zeros(SHAPE, dtype=np.float32)
        gt = np.zeros(SHAPE, dtype=np.float32)
        pred[0, 0:5, 0:20] = 1.0  # 100 voxels
        gt[0, 2:7, 0:20] = 1.0  # 100 voxels, 60 shared

        inter = float((pred * gt).sum())
        expected = 2 * inter / (pred.sum() + gt.sum())
        assert dice_score(pred, gt) == pytest.approx(expected, abs=1e-6)

    def test_prediction_on_empty_ground_truth(self):
        empty = np.zeros(SHAPE, dtype=np.float32)
        assert dice_score(block(), empty) == pytest.approx(0.0, abs=1e-6)

    def test_binarises_non_binary_input(self):
        """Probability-like input is thresholded at >0, not used as weights."""
        soft = block() * 0.4
        assert dice_score(soft, block()) == pytest.approx(1.0, abs=1e-6)


class TestIoU:
    def test_identical_masks(self):
        m = block()
        assert iou_score(m, m) == pytest.approx(1.0, abs=1e-6)

    def test_both_empty_is_one(self):
        empty = np.zeros(SHAPE, dtype=np.float32)
        assert iou_score(empty, empty) == 1.0

    def test_known_value(self):
        """|A|=|B|=100 with 60 shared -> 60 / 140."""
        pred = np.zeros(SHAPE, dtype=np.float32)
        gt = np.zeros(SHAPE, dtype=np.float32)
        pred[0, 0:5, 0:20] = 1.0
        gt[0, 2:7, 0:20] = 1.0

        inter = float((pred * gt).sum())
        union = float(pred.sum() + gt.sum() - inter)
        assert iou_score(pred, gt) == pytest.approx(inter / union, abs=1e-6)

    def test_iou_never_exceeds_dice(self):
        pred = block(z=slice(4, 9))
        gt = block()
        assert iou_score(pred, gt) <= dice_score(pred, gt) + 1e-9


class TestSensitivitySpecificity:
    def test_perfect_recall(self):
        assert sensitivity(block(), block()) == pytest.approx(1.0)

    def test_half_of_ground_truth_found(self):
        gt = block(z=slice(5, 15))  # 10x5x5 = 250 voxels
        pred = block(z=slice(5, 10))  # 5x5x5 = 125 voxels, all inside gt
        assert sensitivity(pred, gt) == pytest.approx(0.5)

    def test_empty_ground_truth_returns_one(self):
        assert sensitivity(block(), np.zeros(SHAPE, dtype=np.float32)) == 1.0

    def test_specificity_penalises_false_positives(self):
        gt = np.zeros(SHAPE, dtype=np.float32)
        pred = block()  # 125 false-positive voxels out of 8000
        expected = (8000 - 125) / 8000
        assert specificity(pred, gt) == pytest.approx(expected)

    def test_specificity_perfect_when_no_false_positives(self):
        m = block()
        assert specificity(m, m) == pytest.approx(1.0)


class TestHausdorff95:
    def test_both_empty_is_zero(self):
        empty = np.zeros(SHAPE, dtype=np.float32)
        assert hausdorff_95(empty, empty) == 0.0

    def test_one_empty_is_infinite(self):
        empty = np.zeros(SHAPE, dtype=np.float32)
        assert hausdorff_95(block(), empty) == float("inf")
        assert hausdorff_95(empty, block()) == float("inf")

    def test_identical_masks_is_zero(self):
        m = block()
        assert hausdorff_95(m, m) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        pred = block(z=slice(4, 9))
        gt = block()
        assert hausdorff_95(pred, gt) == pytest.approx(hausdorff_95(gt, pred), abs=1e-6)

    def test_scales_with_voxel_spacing(self):
        """Doubling the spacing doubles the reported distance in mm."""
        pred = block(z=slice(3, 8))
        gt = block()

        d1 = hausdorff_95(pred, gt, spacing=(1.0, 1.0, 1.0))
        d2 = hausdorff_95(pred, gt, spacing=(2.0, 2.0, 2.0))

        assert d1 > 0
        assert d2 == pytest.approx(2 * d1, rel=1e-6)


class TestVolume:
    def test_isotropic_1mm(self):
        """125 voxels at 1 mm^3 = 0.125 mL."""
        assert volume_ml(block(), (1.0, 1.0, 1.0)) == pytest.approx(0.125)

    def test_anisotropic_spacing(self):
        """125 voxels at 2x2x5 mm = 125 * 20 mm^3 = 2.5 mL."""
        assert volume_ml(block(), (2.0, 2.0, 5.0)) == pytest.approx(2.5)

    def test_empty_mask_is_zero(self):
        assert volume_ml(np.zeros(SHAPE, dtype=np.float32)) == 0.0

    def test_mae_is_absolute_and_symmetric(self):
        big = block(z=slice(5, 15))  # 250 voxels -> 0.25 mL
        small = block()  # 125 voxels -> 0.125 mL

        assert volume_mae(small, big, (1.0, 1.0, 1.0)) == pytest.approx(0.125)
        assert volume_mae(big, small, (1.0, 1.0, 1.0)) == pytest.approx(0.125)


class TestLesionWise:
    def test_both_empty(self):
        empty = np.zeros(SHAPE, dtype=np.float32)
        m = lesion_wise_metrics(empty, empty)
        assert (m["f1"], m["tp"], m["fp"], m["fn"]) == (1.0, 0, 0, 0)

    def test_single_matched_lesion(self):
        m = lesion_wise_metrics(block(), block())
        assert m["tp"] == 1
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["f1"] == pytest.approx(1.0)

    def test_missed_lesion_counts_as_false_negative(self):
        empty = np.zeros(SHAPE, dtype=np.float32)
        m = lesion_wise_metrics(empty, block())
        assert m["fn"] == 1
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_spurious_lesion_counts_as_false_positive(self):
        empty = np.zeros(SHAPE, dtype=np.float32)
        m = lesion_wise_metrics(block(), empty)
        assert m["fp"] == 1
        assert m["precision"] == 0.0

    def test_counts_connected_components_separately(self):
        """Two well-separated ground-truth lesions, only one predicted."""
        gt = np.zeros(SHAPE, dtype=np.float32)
        gt[2:5, 2:5, 2:5] = 1.0
        gt[14:17, 14:17, 14:17] = 1.0

        pred = np.zeros(SHAPE, dtype=np.float32)
        pred[2:5, 2:5, 2:5] = 1.0

        m = lesion_wise_metrics(pred, gt)
        assert m["tp"] == 1
        assert m["fn"] == 1
        assert m["fp"] == 0
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(0.5)

    def test_barely_touching_prediction_is_not_a_match(self):
        """Below the IoU threshold the detection does not count."""
        gt = np.zeros(SHAPE, dtype=np.float32)
        gt[5:15, 5:15, 5:15] = 1.0  # 1000 voxels

        pred = np.zeros(SHAPE, dtype=np.float32)
        pred[14:16, 14:16, 14:16] = 1.0  # 8 voxels, 1 inside gt

        m = lesion_wise_metrics(pred, gt, iou_threshold=0.1)
        assert m["tp"] == 0
        assert m["fp"] == 1
        assert m["fn"] == 1


class TestComputeAllMetrics:
    def test_returns_the_full_key_set(self):
        result = compute_all_metrics(block(), block(), (1.0, 1.0, 1.0))

        assert set(result) == {
            "dice",
            "iou",
            "sensitivity",
            "specificity",
            "hd95",
            "volume_mae_ml",
            "pred_volume_ml",
            "gt_volume_ml",
            "lesion_f1",
            "lesion_precision",
            "lesion_recall",
        }

    def test_all_values_are_plain_floats(self):
        """Results are serialised to JSON by scripts/evaluate.py."""
        result = compute_all_metrics(block(z=slice(4, 9)), block(), (2.0, 2.0, 2.0))
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value).__name__}"

    def test_perfect_prediction(self):
        m = block()
        result = compute_all_metrics(m, m, (1.0, 1.0, 1.0))

        assert result["dice"] == pytest.approx(1.0, abs=1e-6)
        assert result["iou"] == pytest.approx(1.0, abs=1e-6)
        assert result["hd95"] == pytest.approx(0.0, abs=1e-6)
        assert result["volume_mae_ml"] == pytest.approx(0.0)
        assert result["lesion_f1"] == pytest.approx(1.0)
