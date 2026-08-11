"""Tests for V2 CTP perfusion analysis pipeline."""

from __future__ import annotations

import numpy as np


def _make_synthetic_ctp(shape=(64, 64, 40)):
    """Create synthetic CTP data with known core and penumbra."""
    # Normal brain
    tmax = np.random.normal(3.0, 0.5, shape).astype(np.float32)
    tmax = np.clip(tmax, 0.5, 30.0)
    cbf = np.random.normal(50.0, 10.0, shape).astype(np.float32)
    cbf = np.clip(cbf, 5.0, 100.0)
    ncct = np.random.normal(35.0, 5.0, shape).astype(np.float32)

    # Hypoperfusion region (larger)
    tmax[18:38, 18:42, 14:26] = 10.0

    # Core within hypoperfusion (smaller, very low CBF)
    cbf[22:32, 24:36, 16:24] = 5.0

    return tmax, cbf, ncct


class TestBrainMask:
    def test_basic(self):
        from src.v2_perfusion.brain_mask import estimate_brain_mask_ct

        ncct = np.zeros((64, 64, 40), dtype=np.float32)
        ncct[10:50, 10:50, 5:35] = 35.0  # brain tissue ~35 HU
        ncct[5:8, 5:8, 5:8] = 500.0  # bone

        mask = estimate_brain_mask_ct(ncct)
        assert mask.shape == ncct.shape
        assert mask.dtype == np.float32
        assert mask[30, 30, 20] == 1.0  # inside brain
        # Brain region should be majority of mask
        assert mask.sum() > 0


class TestContralateral:
    def test_basic(self):
        from src.v2_perfusion.contralateral import estimate_contralateral_cbf

        cbf = np.full((64, 64, 40), 50.0, dtype=np.float32)
        cbf[20:30, 20:30, 15:25] = 5.0  # low CBF core

        result = estimate_contralateral_cbf(cbf)
        assert result > 0
        assert abs(result - 50.0) < 10.0  # should be near normal

    def test_empty(self):
        from src.v2_perfusion.contralateral import estimate_contralateral_cbf

        cbf = np.zeros((10, 10, 10), dtype=np.float32)
        result = estimate_contralateral_cbf(cbf)
        assert result >= 1.0  # prevent div by zero


class TestThresholdMaps:
    def test_hypoperfusion(self):
        from src.v2_perfusion.threshold_maps import compute_hypoperfusion_mask

        tmax = np.full((64, 64, 40), 3.0, dtype=np.float32)
        tmax[20:30, 20:30, 15:25] = 10.0  # above threshold

        mask = compute_hypoperfusion_mask(tmax, threshold_sec=6.0)
        assert mask.shape == tmax.shape
        assert mask[25, 25, 20] == 1.0  # in hypoperfusion
        assert mask[10, 10, 10] == 0.0  # normal

    def test_core(self):
        from src.v2_perfusion.threshold_maps import compute_core_mask

        cbf = np.full((64, 64, 40), 50.0, dtype=np.float32)
        cbf[22:28, 24:30, 16:22] = 5.0  # very low CBF

        mask = compute_core_mask(cbf, threshold_rcbf=0.30)
        assert mask.shape == cbf.shape
        assert mask[25, 27, 19] == 1.0  # in core
        assert mask[10, 10, 10] == 0.0  # normal

    def test_penumbra(self):
        from src.v2_perfusion.threshold_maps import (
            compute_core_mask,
            compute_hypoperfusion_mask,
            compute_penumbra_mask,
        )

        tmax, cbf, _ = _make_synthetic_ctp()
        hypo = compute_hypoperfusion_mask(tmax, 6.0)
        core = compute_core_mask(cbf, 0.30)
        penumbra = compute_penumbra_mask(hypo, core)

        assert penumbra.shape == tmax.shape
        # Penumbra should be smaller than hypoperfusion
        assert penumbra.sum() <= hypo.sum()
        # Penumbra should not overlap with core
        assert ((penumbra > 0) & (core > 0)).sum() == 0

    def test_mismatch_target(self):
        from src.v2_perfusion.threshold_maps import compute_mismatch_metrics

        # Large hypoperfusion, small core -> target mismatch
        hypo = np.zeros((64, 64, 40), dtype=np.float32)
        hypo[10:50, 10:50, 5:35] = 1.0  # large
        core = np.zeros((64, 64, 40), dtype=np.float32)
        core[25:30, 25:30, 18:22] = 1.0  # small

        result = compute_mismatch_metrics(core, hypo, spacing=(1.0, 1.0, 1.0))
        assert result["target_mismatch_status"] == "target_mismatch"
        assert result["mismatch_ratio"] > 1.8
        assert result["mismatch_volume_ml"] > 15.0

    def test_mismatch_no_target(self):
        from src.v2_perfusion.threshold_maps import compute_mismatch_metrics

        # Similar sized core and hypoperfusion -> no mismatch
        both = np.zeros((64, 64, 40), dtype=np.float32)
        both[20:30, 20:30, 15:25] = 1.0

        result = compute_mismatch_metrics(both, both, spacing=(1.0, 1.0, 1.0))
        assert result["target_mismatch_status"] == "no_mismatch"
        assert result["mismatch_ratio"] <= 1.8


class TestQC:
    def test_pass(self):
        from src.v2_perfusion.qc import check_perfusion_quality

        tmax = np.random.uniform(1, 10, (64, 64, 40)).astype(np.float32)
        cbf = np.random.uniform(10, 80, (64, 64, 40)).astype(np.float32)

        result = check_perfusion_quality({"tmax": tmax, "cbf": cbf})
        assert result["passed"] is True

    def test_missing_map(self):
        from src.v2_perfusion.qc import check_perfusion_quality

        result = check_perfusion_quality({"tmax": np.zeros((10, 10, 10))})
        assert result["passed"] is False
        assert any("cbf" in r for r in result["reasons"])

    def test_nan_fraction(self):
        from src.v2_perfusion.qc import check_perfusion_quality

        tmax = np.full((100, 100, 100), 5.0, dtype=np.float32)
        tmax[:50] = np.nan  # 50% NaN
        cbf = np.full((100, 100, 100), 50.0, dtype=np.float32)

        result = check_perfusion_quality({"tmax": tmax, "cbf": cbf}, max_nan_fraction=0.01)
        assert result["passed"] is False


class TestV2Builder:
    def test_build(self):
        from src.findings.v2_builder import build_v2_findings

        tmax, cbf, _ = _make_synthetic_ctp()
        metadata = {"subject_id": "test", "spacing": (1.0, 1.0, 5.0)}

        findings = build_v2_findings(tmax, cbf, metadata)

        assert findings["perfusion_source"] == "CTP"
        assert findings["maps_present"]["tmax"] is True
        assert findings["maps_present"]["cbf"] is True
        assert findings["core"]["volume_ml"] > 0
        assert findings["hypoperfusion"]["volume_ml"] > 0
        assert findings["mismatch"]["target_mismatch_status"] in (
            "target_mismatch",
            "no_mismatch",
            "indeterminate",
        )
        assert "_masks" in findings  # internal masks for viz

    def test_combined(self):
        from src.findings.v2_builder import build_combined_findings

        v1 = {"study_id": "test", "lesions": [], "total_lesion_count": 0}
        v2 = {"perfusion_source": "CTP", "core": {"volume_ml": 5.0}, "_masks": {}}

        combined = build_combined_findings(v1, v2)
        assert "perfusion" in combined
        assert combined["perfusion"]["perfusion_source"] == "CTP"
        assert "_masks" not in combined["perfusion"]  # cleaned


class TestReportWithPerfusion:
    def test_generate_with_perfusion(self):
        from src.report.generator import generate_report

        findings = {
            "study_id": "test-ctp",
            "model_version": "v2.0-ctp-threshold",
            "timestamp": "2026-03-31T00:00:00+00:00",
            "protocol_present": {"dwi": False, "adc": False, "flair": False},
            "quality_gate": {"passed": True, "reasons": []},
            "lesions": [],
            "total_lesion_count": 0,
            "total_lesion_volume_ml": 0.0,
            "overall_impression": "indeterminate",
            "overall_confidence": 0.5,
            "combined_mask_ref": "",
            "perfusion": {
                "perfusion_source": "CTP",
                "maps_present": {"tmax": True, "cbf": True, "cbv": False, "mtt": False},
                "thresholds_used": {
                    "hypoperfusion_tmax_sec": 6.0,
                    "core_rcbf_threshold": 0.30,
                },
                "perfusion_quality_gate": {"passed": True, "reasons": []},
                "core": {
                    "volume_ml": 15.0,
                    "mask_ref": "core.nii.gz",
                    "method": "rCBF_threshold",
                    "confidence": 0.8,
                },
                "hypoperfusion": {
                    "volume_ml": 80.0,
                    "mask_ref": "hypo.nii.gz",
                    "method": "Tmax_threshold",
                    "confidence": 0.8,
                },
                "penumbra": {"volume_ml": 65.0, "mask_ref": "penumbra.nii.gz"},
                "mismatch": {
                    "mismatch_volume_ml": 65.0,
                    "mismatch_ratio": 5.3,
                    "target_mismatch_status": "target_mismatch",
                    "criteria_used": {
                        "core_max_ml": 70,
                        "mismatch_ratio_min": 1.8,
                        "mismatch_volume_min_ml": 15,
                    },
                    "confidence": 0.85,
                },
            },
        }

        report = generate_report(findings)
        assert "CT PERFUSION ANALYSIS" in report
        assert "15.0" in report  # core volume
        assert "80.0" in report  # hypoperfusion volume
        assert "TARGET MISMATCH" in report
        assert "6.0" in report  # Tmax threshold

    def test_validate_with_perfusion(self):
        from src.report.generator import generate_report
        from src.report.validator import validate_report

        findings = {
            "study_id": "test-ctp",
            "model_version": "v2.0",
            "timestamp": "2026-03-31T00:00:00",
            "protocol_present": {"dwi": False, "adc": False, "flair": False},
            "quality_gate": {"passed": True, "reasons": []},
            "lesions": [],
            "total_lesion_count": 0,
            "total_lesion_volume_ml": 0.0,
            "overall_impression": "indeterminate",
            "overall_confidence": 0.5,
            "combined_mask_ref": "",
            "perfusion": {
                "perfusion_source": "CTP",
                "maps_present": {"tmax": True, "cbf": True, "cbv": False, "mtt": False},
                "thresholds_used": {"hypoperfusion_tmax_sec": 6.0, "core_rcbf_threshold": 0.30},
                "perfusion_quality_gate": {"passed": True, "reasons": []},
                "core": {
                    "volume_ml": 10.0,
                    "mask_ref": "c.nii.gz",
                    "method": "rCBF",
                    "confidence": 0.8,
                },
                "hypoperfusion": {
                    "volume_ml": 50.0,
                    "mask_ref": "h.nii.gz",
                    "method": "Tmax",
                    "confidence": 0.8,
                },
                "penumbra": {"volume_ml": 40.0, "mask_ref": "p.nii.gz"},
                "mismatch": {
                    "mismatch_volume_ml": 40.0,
                    "mismatch_ratio": 5.0,
                    "target_mismatch_status": "target_mismatch",
                    "criteria_used": {
                        "core_max_ml": 70,
                        "mismatch_ratio_min": 1.8,
                        "mismatch_volume_min_ml": 15,
                    },
                    "confidence": 0.85,
                },
            },
        }

        report = generate_report(findings)
        validation = validate_report(report, findings)
        assert validation["valid"] is True, f"Issues: {validation['issues']}"
