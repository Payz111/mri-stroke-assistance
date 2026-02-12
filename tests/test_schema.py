"""Tests for Pydantic schema validation."""

from __future__ import annotations

import pytest

from src.findings.schema import (
    EvidenceSlices,
    Lesion,
    ProtocolPresent,
    QualityGate,
    V1Findings,
)


class TestV1Findings:
    def test_valid_findings(self):
        """A well-formed V1Findings should validate."""
        findings = V1Findings(
            study_id="sub-001",
            model_version="v0.1.0",
            timestamp="2026-02-11T12:00:00Z",
            protocol_present=ProtocolPresent(dwi=True, adc=True, flair=True),
            quality_gate=QualityGate(passed=True),
            lesions=[],
            total_lesion_count=0,
            total_lesion_volume_ml=0.0,
            overall_impression="no_acute_infarct",
            overall_confidence=0.95,
            combined_mask_ref="combined_mask.nii.gz",
        )
        assert findings.total_lesion_count == 0

    def test_lesion_validation(self):
        """A Lesion with valid fields should validate."""
        lesion = Lesion(
            lesion_id=1,
            mask_ref="lesion_001.nii.gz",
            volume_ml=15.3,
            max_diameter_mm=32.5,
            laterality="left",
            anatomic_location=["frontal", "parietal"],
            vascular_territory="MCA",
            territory_confidence=0.92,
            adc_low_confirmed=True,
            adc_confidence=0.95,
            flair_hyperintensity="subtle",
            flair_confidence=0.78,
            dwi_flair_mismatch="yes",
            mismatch_confidence=0.85,
            evidence_slices=EvidenceSlices(slice_indices=[45, 46, 47]),
        )
        assert lesion.volume_ml == 15.3

    def test_negative_volume_fails(self):
        """Volume must be >= 0."""
        with pytest.raises(Exception):
            Lesion(
                lesion_id=1,
                mask_ref="x.nii.gz",
                volume_ml=-1.0,
                max_diameter_mm=10.0,
                laterality="left",
                anatomic_location=["frontal"],
                vascular_territory="MCA",
                territory_confidence=0.5,
                adc_low_confirmed=True,
                adc_confidence=0.5,
                flair_hyperintensity="none",
                flair_confidence=0.5,
                dwi_flair_mismatch="no",
                mismatch_confidence=0.5,
                evidence_slices=EvidenceSlices(),
            )
