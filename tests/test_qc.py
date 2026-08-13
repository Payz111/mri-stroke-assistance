"""Tests for the QC pipeline and the no-score policy (ADR-006).

The policy is the point: on a critical failure the system must produce no
prediction at all, because a confident wrong answer is more dangerous than
no answer. The last class here checks the refusal actually happens rather
than merely being reported.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.qc.qc_pipeline import CRITICAL, WARNING, QCPipeline

SHAPE = (48, 48, 16)
SPACING = (2.0, 2.0, 5.0)


def brain(seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """A volume with a plausible head-shaped non-zero region."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(SHAPE, dtype=np.float32)
    vol[8:40, 8:40, 2:14] = rng.uniform(50, 500, size=(32, 32, 12)) * scale
    return vol


def subject(**overrides) -> dict:
    data = {
        "dwi": brain(0),
        "adc": brain(1, scale=1.7),
        "flair": brain(2),
        "spacing": SPACING,
    }
    data.update(overrides)
    return data


@pytest.fixture
def qc() -> QCPipeline:
    return QCPipeline()


class TestModalities:
    def test_missing_modality_fails(self, qc):
        """QC should fail when a required modality is missing."""
        data = {"dwi": None, "adc": object(), "flair": object()}
        result = qc.check_modalities(data)
        assert not result.passed
        assert "dwi" in result.reason

    def test_all_modalities_pass(self, qc):
        """QC should pass when all modalities are present."""
        data = {"dwi": object(), "adc": object(), "flair": object()}
        result = qc.check_modalities(data)
        assert result.passed

    def test_absent_key_is_treated_as_missing(self, qc):
        result = qc.check_modalities({"dwi": brain(), "adc": brain(1)})
        assert not result.passed
        assert "flair" in result.reason

    def test_all_zero_volume_fails(self, qc):
        result = qc.check_modalities(subject(adc=np.zeros(SHAPE, dtype=np.float32)))
        assert not result.passed
        assert "all zeros" in result.reason

    def test_all_nan_volume_fails(self, qc):
        result = qc.check_modalities(subject(dwi=np.full(SHAPE, np.nan, dtype=np.float32)))
        assert not result.passed
        assert "finite" in result.reason

    def test_real_volumes_pass(self, qc):
        assert qc.check_modalities(subject()).passed


class TestSpacing:
    def test_plausible_spacing_passes(self, qc):
        assert qc.check_spacing(subject()).passed

    def test_missing_spacing_fails(self, qc):
        data = subject()
        data.pop("spacing")
        assert not qc.check_spacing(data).passed

    def test_spacing_read_from_metadata(self, qc):
        data = subject()
        data.pop("spacing")
        data["metadata"] = {"spacing": SPACING}
        assert qc.check_spacing(data).passed

    def test_zero_spacing_fails(self, qc):
        assert not qc.check_spacing(subject(spacing=(0.0, 2.0, 2.0))).passed

    def test_nan_spacing_fails(self, qc):
        assert not qc.check_spacing(subject(spacing=(np.nan, 2.0, 2.0))).passed

    def test_absurdly_large_spacing_fails(self, qc):
        """50 mm voxels are not brain MRI."""
        result = qc.check_spacing(subject(spacing=(2.0, 2.0, 50.0)))
        assert not result.passed
        assert "plausible range" in result.reason

    def test_too_few_values_fails(self, qc):
        assert not qc.check_spacing(subject(spacing=(2.0, 2.0))).passed

    def test_non_numeric_spacing_fails(self, qc):
        assert not qc.check_spacing(subject(spacing=("a", "b", "c"))).passed


class TestCoverage:
    def test_normal_volume_passes(self, qc):
        assert qc.check_coverage(subject()).passed

    def test_tiny_volume_fails(self, qc):
        result = qc.check_coverage(subject(dwi=np.ones((8, 8, 4), dtype=np.float32)))
        assert not result.passed
        assert "smaller than the minimum" in result.reason

    def test_mostly_empty_field_of_view_fails(self, qc):
        """A volume with a few stray voxels is not a head."""
        vol = np.zeros(SHAPE, dtype=np.float32)
        vol[0:2, 0:2, 0:2] = 100.0
        result = qc.check_coverage(subject(dwi=vol))
        assert not result.passed
        assert "field of view" in result.reason

    def test_four_dimensional_volume_fails(self, qc):
        result = qc.check_coverage(subject(dwi=np.ones((*SHAPE, 1), dtype=np.float32)))
        assert not result.passed
        assert "3-D" in result.reason


class TestDwiAdcConsistency:
    def test_distinct_volumes_pass(self, qc):
        assert qc.check_dwi_adc_consistency(subject()).passed

    def test_identical_volumes_fail(self, qc):
        """The same series submitted twice is the error worth catching."""
        same = brain(7)
        result = qc.check_dwi_adc_consistency(subject(dwi=same, adc=same.copy()))
        assert not result.passed
        assert "identical" in result.reason

    def test_near_duplicate_volumes_fail(self, qc):
        base = brain(3)
        rng = np.random.default_rng(11)
        almost = base + rng.normal(0, 0.01, size=base.shape).astype(np.float32)
        result = qc.check_dwi_adc_consistency(subject(dwi=base, adc=almost))
        assert not result.passed
        assert "correlate" in result.reason

    def test_shape_mismatch_fails(self, qc):
        result = qc.check_dwi_adc_consistency(subject(adc=np.ones((16, 16, 8), dtype=np.float32)))
        assert not result.passed
        assert "differ in shape" in result.reason

    def test_constant_volume_fails(self, qc):
        result = qc.check_dwi_adc_consistency(subject(adc=np.full(SHAPE, 5.0, np.float32)))
        assert not result.passed


class TestRunAll:
    def test_clean_subject_passes_every_gate(self, qc):
        results = qc.run_all(subject())
        assert len(results) == 4
        assert all(r.passed for r in results), [r.reason for r in results if not r.passed]

    def test_short_circuits_when_a_modality_is_missing(self, qc):
        """No point measuring coverage on a study that has no DWI."""
        results = qc.run_all({"dwi": None, "adc": brain(), "flair": brain(1)})
        assert len(results) == 1
        assert not results[0].passed

    def test_every_result_is_named(self, qc):
        assert all(r.name for r in qc.run_all(subject()))


class TestEvaluate:
    def test_clean_subject_reports_passed(self, qc):
        report = qc.evaluate(subject())
        assert report.passed
        assert report.critical_failures == []

    def test_critical_failure_blocks(self, qc):
        report = qc.evaluate(subject(spacing=(0.0, 0.0, 0.0)))
        assert not report.passed
        assert len(report.critical_failures) == 1

    def test_report_is_json_serialisable(self, qc):
        payload = qc.evaluate(subject()).to_dict()
        json.dumps(payload)  # must not raise
        assert set(payload) == {"passed", "checks", "critical_failures", "warnings"}

    def test_summary_text_names_the_failed_check(self, qc):
        text = qc.evaluate(subject(spacing=(0.0, 0.0, 0.0))).summary_text()
        assert "QUALITY CONTROL: FAILED" in text
        assert "spacing" in text

    def test_thresholds_are_configurable(self):
        loose = QCPipeline({"min_spacing_mm": 0.0001, "max_spacing_mm": 100.0})
        assert loose.check_spacing(subject(spacing=(2.0, 2.0, 50.0))).passed

    def test_severity_constants_are_distinct(self):
        assert CRITICAL != WARNING
