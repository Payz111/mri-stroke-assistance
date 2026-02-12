"""Tests for QC pipeline."""

from __future__ import annotations

import pytest


class TestQCPipeline:
    def test_missing_modality_fails(self):
        """QC should fail when a required modality is missing."""
        from src.qc.qc_pipeline import QCPipeline

        qc = QCPipeline()
        data = {"dwi": None, "adc": object(), "flair": object()}
        result = qc.check_modalities(data)
        assert not result.passed

    def test_all_modalities_pass(self):
        """QC should pass when all modalities are present."""
        from src.qc.qc_pipeline import QCPipeline

        qc = QCPipeline()
        data = {"dwi": object(), "adc": object(), "flair": object()}
        result = qc.check_modalities(data)
        assert result.passed
