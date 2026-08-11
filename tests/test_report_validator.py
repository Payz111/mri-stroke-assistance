"""Tests for report validator (zero hallucinations)."""

from __future__ import annotations

import numpy as np
import pytest

from src.findings.builder import build_findings
from src.report.generator import generate_report
from src.report.validator import validate_report

SHAPE = (64, 64, 40)
METADATA = {"subject_id": "sub-test-001", "spacing": (1.0, 1.0, 3.0)}


def _synthetic_volumes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build DWI/ADC/FLAIR volumes with an acute-ischemia signal pattern."""
    rng = np.random.default_rng(42)
    dwi = (rng.standard_normal(SHAPE) * 100 + 500).astype(np.float32)
    adc = (rng.standard_normal(SHAPE) * 200 + 800).astype(np.float32)
    flair = (rng.standard_normal(SHAPE) * 100 + 400).astype(np.float32)

    # Acute pattern: DWI bright, ADC dark, FLAIR still normal (mismatch positive)
    dwi[mask > 0] = 1200.0
    adc[mask > 0] = 300.0
    flair[mask > 0] = 450.0
    return dwi, adc, flair


def _build(mask: np.ndarray) -> tuple[dict, str]:
    """Run the findings -> report chain for *mask*."""
    dwi, adc, flair = _synthetic_volumes(mask)
    findings = build_findings(
        lesion_masks=[mask],
        dwi=dwi,
        adc=adc,
        flair=flair,
        metadata=METADATA,
    )
    return findings, generate_report(findings)


@pytest.fixture
def lesion_case() -> tuple[dict, str]:
    """A single left-hemisphere lesion of ~1000 voxels."""
    mask = np.zeros(SHAPE, dtype=np.float32)
    mask[20:30, 25:35, 15:25] = 1.0
    return _build(mask)


@pytest.fixture
def empty_case() -> tuple[dict, str]:
    """A study with no lesion at all."""
    return _build(np.zeros(SHAPE, dtype=np.float32))


class TestReportValidator:
    def test_valid_report_passes(self, lesion_case):
        """A report grounded in findings should pass validation."""
        findings, report = lesion_case

        result = validate_report(report, findings)

        assert result["valid"], f"Unexpected issues: {result['issues']}"
        assert result["issues"] == []
        assert all(result["checks"].values())

    def test_hallucinated_claim_fails(self, lesion_case):
        """A report with numbers absent from findings should fail."""
        findings, report = lesion_case

        tampered = report + "\nAdditional lesion of 42.7 mL in the right frontal lobe.\n"
        result = validate_report(tampered, findings)

        assert not result["valid"]
        assert result["checks"]["no_hallucinated_numbers"] is False
        assert any("42.7" in issue for issue in result["issues"])

    def test_wrong_volume_fails(self, lesion_case):
        """A report whose stated volume contradicts findings should fail."""
        findings, report = lesion_case
        true_volume = f"{findings['total_lesion_volume_ml']:.1f}"

        tampered = report.replace(true_volume, "99.9")
        result = validate_report(tampered, findings)

        assert not result["valid"]
        assert result["checks"]["total_volume"] is False

    def test_empty_report_passes(self, empty_case):
        """A study with no lesions has nothing to hallucinate."""
        findings, report = empty_case

        assert findings["total_lesion_count"] == 0

        result = validate_report(report, findings)

        assert result["valid"], f"Unexpected issues: {result['issues']}"
