"""Tests for the end-to-end inference pipeline.

These exercise the plumbing (preprocess -> model -> findings -> report ->
validation) with an untrained tiny model. They assert on shapes and contract,
never on segmentation quality -- accuracy is covered by scripts/evaluate.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.pipeline import predict_mask, run_inference
from src.models.factory import create_model

SHAPE = (64, 64, 40)
METADATA = {"subject_id": "sub-test-001", "spacing": (1.0, 1.0, 3.0)}

TINY_MODEL_CONFIG = {
    "name": "unet3d",
    "in_channels": 3,
    "out_channels": 1,
    "features": [4, 8, 16],
    "dropout": 0.0,
}


@pytest.fixture(scope="module")
def model():
    """An untrained tiny UNet3D -- enough to validate the pipeline contract."""
    net = create_model(TINY_MODEL_CONFIG)
    net.eval()
    return net


@pytest.fixture
def volumes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    dwi = (rng.standard_normal(SHAPE) * 100 + 500).astype(np.float32)
    adc = (rng.standard_normal(SHAPE) * 200 + 800).astype(np.float32)
    flair = (rng.standard_normal(SHAPE) * 100 + 400).astype(np.float32)
    return dwi, adc, flair


class TestPredictMask:
    def test_mask_matches_input_shape(self, model, volumes):
        """Prediction is cropped back to the original DWI geometry."""
        dwi, adc, flair = volumes

        mask = predict_mask(model, dwi, adc, flair)

        assert mask.shape == dwi.shape

    def test_mask_is_binary(self, model, volumes):
        """Thresholding must yield strictly 0/1 values."""
        dwi, adc, flair = volumes

        mask = predict_mask(model, dwi, adc, flair)

        assert set(np.unique(mask)).issubset({0.0, 1.0})


class TestInferencePipeline:
    def test_pipeline_returns_dict(self, model, volumes):
        """Inference pipeline should return a results dict."""
        dwi, adc, flair = volumes

        result = run_inference(model, dwi, adc, flair, METADATA)

        assert isinstance(result, dict)

    def test_output_contains_required_keys(self, model, volumes):
        """Output must contain mask, findings, report, and validation."""
        dwi, adc, flair = volumes

        result = run_inference(model, dwi, adc, flair, METADATA)

        assert set(result) == {"pred_mask", "findings", "report", "validation"}
        assert result["pred_mask"].shape == dwi.shape
        assert isinstance(result["report"], str) and result["report"]
        assert "total_lesion_count" in result["findings"]
        assert "valid" in result["validation"]

    def test_handles_flair_with_different_resolution(self, model, volumes):
        """FLAIR is routinely acquired at a different resolution than DWI."""
        dwi, adc, _ = volumes
        rng = np.random.default_rng(1)
        flair_hires = (rng.standard_normal((128, 128, 60)) * 100 + 400).astype(np.float32)

        result = run_inference(model, dwi, adc, flair_hires, METADATA)

        assert result["pred_mask"].shape == dwi.shape
