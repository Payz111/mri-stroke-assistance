"""Tests for prediction post-processing.

`remove_small_components` is applied by scripts/evaluate.py before metrics are
computed, so it directly shapes the published lesion-F1 figure.
"""

from __future__ import annotations

import numpy as np

from src.eval.postprocess import remove_small_components

SHAPE = (20, 20, 20)


def _empty() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.float32)


class TestRemoveSmallComponents:
    def test_empty_mask_stays_empty(self):
        out = remove_small_components(_empty(), min_size=10)
        assert out.sum() == 0
        assert out.shape == SHAPE

    def test_large_component_is_kept(self):
        mask = _empty()
        mask[5:10, 5:10, 5:10] = 1.0  # 125 voxels

        out = remove_small_components(mask, min_size=10)

        assert out.sum() == 125

    def test_small_component_is_dropped(self):
        mask = _empty()
        mask[5:7, 5:7, 5] = 1.0  # 4 voxels

        out = remove_small_components(mask, min_size=10)

        assert out.sum() == 0

    def test_keeps_large_drops_small_in_the_same_mask(self):
        mask = _empty()
        mask[2:8, 2:8, 2:8] = 1.0  # 216 voxels, kept
        mask[15, 15, 15] = 1.0  # 1 voxel, dropped

        out = remove_small_components(mask, min_size=10)

        assert out.sum() == 216
        assert out[15, 15, 15] == 0
        assert out[2, 2, 2] == 1

    def test_component_of_exactly_min_size_is_kept(self):
        """The threshold is inclusive: >= min_size survives."""
        mask = _empty()
        mask[5, 5, 0:10] = 1.0  # exactly 10 voxels

        out = remove_small_components(mask, min_size=10)

        assert out.sum() == 10

    def test_component_one_below_min_size_is_dropped(self):
        mask = _empty()
        mask[5, 5, 0:9] = 1.0  # 9 voxels

        out = remove_small_components(mask, min_size=10)

        assert out.sum() == 0

    def test_min_size_one_keeps_everything(self):
        mask = _empty()
        mask[1, 1, 1] = 1.0
        mask[10:12, 10:12, 10:12] = 1.0

        out = remove_small_components(mask, min_size=1)

        assert out.sum() == 1 + 8

    def test_output_is_binary_float32(self):
        mask = _empty()
        mask[5:10, 5:10, 5:10] = 0.7  # non-binary input

        out = remove_small_components(mask, min_size=10)

        assert out.dtype == np.float32
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_input_is_not_mutated(self):
        mask = _empty()
        mask[15, 15, 15] = 1.0
        before = mask.copy()

        remove_small_components(mask, min_size=10)

        assert np.array_equal(mask, before)

    def test_diagonally_touching_voxels_are_separate_components(self):
        """scipy.ndimage.label defaults to face connectivity in 3-D."""
        mask = _empty()
        mask[5, 5, 5] = 1.0
        mask[6, 6, 6] = 1.0  # touches only at a corner

        out = remove_small_components(mask, min_size=2)

        assert out.sum() == 0
