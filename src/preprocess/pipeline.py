"""End-to-end preprocessing pipeline for MRI stroke data.

Orchestrates per-subject and batch preprocessing steps including
registration, skull-stripping, intensity normalisation, and resampling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def preprocess_subject(subject_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Run the full preprocessing pipeline for a single subject.

    Parameters
    ----------
    subject_path:
        Path to the raw subject directory (expected to contain DWI, ADC,
        FLAIR NIfTI files).
    config:
        Preprocessing configuration (target spacing, normalisation mode, …).

    Returns
    -------
    dict
        Preprocessed volumes and metadata keyed by modality name.
    """
    raise NotImplementedError()


def preprocess_dataset(
    raw_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """Batch-preprocess every subject in *raw_dir* and write results.

    Parameters
    ----------
    raw_dir:
        Root directory containing one sub-folder per subject.
    output_dir:
        Destination directory for preprocessed NIfTI files.
    config:
        Preprocessing configuration dictionary.
    """
    raise NotImplementedError()
