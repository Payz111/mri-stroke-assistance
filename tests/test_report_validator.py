"""Tests for report validator (zero hallucinations)."""

from __future__ import annotations

import pytest


class TestReportValidator:
    def test_valid_report_passes(self):
        """A report grounded in findings should pass validation."""
        pytest.skip("Report validator not yet implemented")

    def test_hallucinated_claim_fails(self):
        """A report with claims not in findings should fail."""
        pytest.skip("Report validator not yet implemented")

    def test_empty_report_passes(self):
        """An empty report has no hallucinations."""
        pytest.skip("Report validator not yet implemented")
