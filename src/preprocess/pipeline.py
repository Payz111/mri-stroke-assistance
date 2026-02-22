"""End-to-end preprocessing pipeline for MRI stroke data.

Orchestrates per-subject preprocessing steps including loading,
registration, intensity normalisation, and resampling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.preprocess.intensity_norm import normalize_zscore
from src.preprocess.registration import resample_to_reference

logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load a NIfTI file and return (data, affine, spacing)."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, affine, spacing


def preprocess_subject(
    subject_path: Path,
    derivatives_path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full preprocessing pipeline for a single subject.

    Parameters
    ----------
    subject_path:
        Path to the raw subject directory (e.g. data_root/sub-strokecaseXXXX).
    derivatives_path:
        Path to the derivatives directory for this subject.
    config:
        Preprocessing configuration. Supported keys:
        - normalize (bool): apply z-score normalization (default True)
        - resample (bool): resample all modalities to DWI space (default True)

    Returns
    -------
    dict
        Preprocessed volumes and metadata.
    """
    if config is None:
        config = {}

    do_normalize = config.get("normalize", True)
    do_resample = config.get("resample", True)

    ses_path = subject_path / "ses-0001"

    # Resolve file paths
    dwi_path = next(ses_path.glob("dwi/*dwi.nii.gz"))
    adc_path = next(ses_path.glob("dwi/*adc.nii.gz"))
    flair_path = next(ses_path.glob("anat/*FLAIR.nii.gz"))

    deriv_ses = derivatives_path / "ses-0001"
    mask_path = next(deriv_ses.glob("*msk.nii.gz"))

    # Load volumes
    dwi, dwi_affine, dwi_spacing = load_nifti(dwi_path)
    adc, _, _ = load_nifti(adc_path)
    flair, _, flair_spacing = load_nifti(flair_path)
    mask, _, _ = load_nifti(mask_path)

    # Binarize mask
    mask = (mask > 0).astype(np.float32)

    # Resample FLAIR (and potentially ADC) to DWI space
    if do_resample:
        if flair.shape != dwi.shape:
            logger.info("Resampling FLAIR %s -> %s", flair.shape, dwi.shape)
            flair = resample_to_reference(flair, dwi, order=1)
        if adc.shape != dwi.shape:
            logger.info("Resampling ADC %s -> %s", adc.shape, dwi.shape)
            adc = resample_to_reference(adc, dwi, order=1)

    # Normalize intensities
    if do_normalize:
        dwi = normalize_zscore(dwi)
        adc = normalize_zscore(adc)
        flair = normalize_zscore(flair)

    return {
        "dwi": dwi.astype(np.float32),
        "adc": adc.astype(np.float32),
        "flair": flair.astype(np.float32),
        "mask": mask,
        "metadata": {
            "subject_id": subject_path.name,
            "spacing": dwi_spacing,
            "affine": dwi_affine,
            "shape": dwi.shape,
        },
    }


def preprocess_dataset(
    data_root: Path,
    derivatives_root: Path,
    output_dir: Path,
    split_file: Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Batch-preprocess subjects and save as .npz files.

    Parameters
    ----------
    data_root:
        Root directory containing subject folders.
    derivatives_root:
        Root directory containing derivative (mask) folders.
    output_dir:
        Destination directory for preprocessed .npz files.
    split_file:
        Optional path to split JSON. If provided, only processes
        subjects in 'train' and 'val' lists.
    config:
        Preprocessing configuration dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine subject list
    if split_file is not None:
        with open(split_file) as f:
            split_data = json.load(f)
        subject_ids = split_data.get("train", []) + split_data.get("val", [])
    else:
        subject_ids = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        )

    logger.info("Preprocessing %d subjects...", len(subject_ids))

    for i, sid in enumerate(subject_ids):
        logger.info("[%d/%d] %s", i + 1, len(subject_ids), sid)

        try:
            result = preprocess_subject(
                subject_path=data_root / sid,
                derivatives_path=derivatives_root / sid,
                config=config,
            )

            out_path = output_dir / f"{sid}.npz"
            np.savez_compressed(
                out_path,
                dwi=result["dwi"],
                adc=result["adc"],
                flair=result["flair"],
                mask=result["mask"],
            )
        except Exception:
            logger.exception("Failed to process %s", sid)
            continue

    logger.info("Done. Output saved to %s", output_dir)
