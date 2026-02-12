"""Report generation — structured findings to prose.

Converts the V1Findings dictionary into a human-readable radiology
report string using the templates defined in :mod:`src.report.templates`.
"""
from __future__ import annotations

from typing import Any


def generate_report(findings: dict[str, Any]) -> str:
    """Generate a textual radiology report from structured findings.

    Parameters
    ----------
    findings:
        V1Findings dictionary as produced by
        :func:`src.findings.builder.build_findings`.

    Returns
    -------
    str
        Multi-section radiology report text (Findings + Impression).
    """
    raise NotImplementedError()
