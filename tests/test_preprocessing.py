"""Tests for preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest


class TestIntensityNorm:
    def test_zscore_output_shape(self):
        """Z-score normalization preserves shape."""
        from src.preprocess.intensity_norm import normalize_zscore

        volume = np.random.randn(64, 64, 32).astype(np.float32)
        result = normalize_zscore(volume)
        assert result.shape == volume.shape

    def test_zscore_zero_mean(self):
        """Z-score normalization produces ~zero mean."""
        from src.preprocess.intensity_norm import normalize_zscore

        volume = np.random.randn(64, 64, 32).astype(np.float32) * 100 + 500
        result = normalize_zscore(volume)
        assert abs(result.mean()) < 0.1

    def test_percentile_range(self):
        """Percentile normalization clips to [0, 1]."""
        from src.preprocess.intensity_norm import normalize_percentile

        volume = np.random.randn(64, 64, 32).astype(np.float32)
        result = normalize_percentile(volume)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
