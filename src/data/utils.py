"""Low-level data-loading and split-management utilities.

Provides helpers for reading NIfTI volumes and for loading pre-defined
cross-validation splits from disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_nifti(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a NIfTI file and return its data array plus metadata.

    Parameters
    ----------
    path:
        Path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(volume, metadata)`` where *metadata* includes at least
        ``affine``, ``spacing``, and ``header`` information.
    """
    raise NotImplementedError()


def load_splits(splits_dir: Path, fold: int) -> dict[str, list[str]]:
    """Load the train / validation subject IDs for a given CV fold.

    Parameters
    ----------
    splits_dir:
        Directory containing split definition files (e.g. JSON or CSV).
    fold:
        Zero-based fold index.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...]}`` with subject-ID strings.
    """
    raise NotImplementedError()
