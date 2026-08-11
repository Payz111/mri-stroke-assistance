"""Tests for QC pipeline.

The QC gate is specified but not yet implemented -- ``QCPipeline`` raises
``NotImplementedError``. These tests describe the intended contract and are
skipped until the implementation lands; do not delete them.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="QCPipeline is a stub (src/qc/qc_pipeline.py)")


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
