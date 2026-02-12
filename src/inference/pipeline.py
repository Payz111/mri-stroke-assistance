"""End-to-end inference pipeline.

Takes a raw NIfTI input, preprocesses it, runs the segmentation model,
applies post-processing, builds findings, and generates a report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_inference(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full inference pipeline on a single subject.

    Parameters
    ----------
    input_path:
        Path to the subject directory or a single NIfTI file.
    output_path:
        Directory where outputs (masks, findings JSON, report) are saved.
    config:
        Pipeline configuration (model checkpoint, thresholds, etc.).

    Returns
    -------
    dict
        Summary including paths to saved artefacts, findings JSON, and
        the generated report text.
    """
    raise NotImplementedError()
