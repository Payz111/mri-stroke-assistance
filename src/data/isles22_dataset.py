"""PyTorch Dataset for the ISLES'22 ischemic-stroke segmentation challenge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


def _find_nifti(base: Path, keyword: str) -> Path | None:
    """Find a NIfTI file containing *keyword* in its name under *base*.

    Searches recursively, returns the first real non-empty file.
    Works with any naming convention (BIDS, Kaggle, skull-stripped, etc.)

    Examples of matched files for keyword="dwi":
        - sub-strokecase0001_ses-0001_dwi.nii.gz   (local BIDS)
        - dwi_skull_stripped.nii                     (Kaggle, inside subdir)
        - sub-Stroke43_iso_dwi_skull_stripped.nii    (Kaggle, alt naming)
    """
    kw = keyword.lower()

    # 1. Try .nii.gz first (local / compressed)
    for p in base.rglob("*.nii.gz"):
        if kw in p.name.lower() and p.is_file() and p.stat().st_size > 0:
            return p

    # 2. Fall back to .nii (Kaggle / uncompressed)
    for p in base.rglob("*.nii"):
        if kw in p.name.lower() and p.is_file() and p.stat().st_size > 0:
            return p

    return None


class ISLES22Dataset(Dataset):
    """Dataset class that serves ISLES'22 subjects.

    Each item is a dictionary with the keys:
        - ``dwi``      : DWI volume (D, H, W) float32
        - ``adc``      : ADC volume (D, H, W) float32
        - ``flair``    : FLAIR volume (D, H, W) float32
        - ``mask``     : Ground-truth lesion mask (D, H, W) float32
        - ``metadata`` : dict with subject_id, spacing, affine, shape
    """

    def __init__(
        self,
        data_root: str | Path,
        derivatives_root: str | Path,
        split_file: str | Path | None,
        split: str = "train",
        transform: Any | None = None,
    ) -> None:
        """Initialise the dataset.

        Parameters
        ----------
        split_file:
            Path to a fold JSON. ``None`` enumerates every subject on disk,
            which is what the volume cache needs -- the cache is shared by all
            folds, so it must not be built from one fold's split.
        split:
            Key to read from the split file ("train" or "val"). Ignored when
            *split_file* is None.
        """
        self.data_root = Path(data_root)
        self.derivatives_root = Path(derivatives_root)
        self.transform = transform

        if split_file is None:
            self.subject_ids = sorted(
                d.name for d in self.data_root.iterdir() if d.name.startswith("sub-")
            )
            if not self.subject_ids:
                raise FileNotFoundError(f"No sub-* directories under {self.data_root}")
        else:
            with open(split_file) as f:
                split_data = json.load(f)
            self.subject_ids = split_data[split]

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _get_paths(self, subject_id: str) -> dict[str, Path]:
        """Resolve file paths for a subject.

        Handles both local BIDS (.nii.gz) and Kaggle (.nii in subdirs).
        """
        sub_path = self.data_root / subject_id / "ses-0001"
        deriv_path = self.derivatives_root / subject_id / "ses-0001"

        dwi = _find_nifti(sub_path / "dwi", "dwi")
        adc = _find_nifti(sub_path / "dwi", "adc")
        flair = _find_nifti(sub_path / "anat", "flair")
        mask = _find_nifti(deriv_path, "msk")

        missing = []
        if dwi is None:
            missing.append("DWI")
        if adc is None:
            missing.append("ADC")
        if flair is None:
            missing.append("FLAIR")
        if mask is None:
            missing.append("mask")

        if missing:
            raise FileNotFoundError(f"Missing files for {subject_id}: {missing}")

        return {"dwi": dwi, "adc": adc, "flair": flair, "mask": mask}

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_id = self.subject_ids[index]
        paths = self._get_paths(subject_id)

        # Load NIfTI volumes
        dwi_img = nib.load(paths["dwi"])
        adc_img = nib.load(paths["adc"])
        flair_img = nib.load(paths["flair"])
        mask_img = nib.load(paths["mask"])

        # Get data as float32 numpy arrays
        dwi = dwi_img.get_fdata(dtype=np.float32)
        adc = adc_img.get_fdata(dtype=np.float32)
        flair = flair_img.get_fdata(dtype=np.float32)
        mask = mask_img.get_fdata(dtype=np.float32)

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        metadata = {
            "subject_id": subject_id,
            "dataset": "isles22",
            "spacing": dwi_img.header.get_zooms()[:3],
            "affine": dwi_img.affine,
            "shape": dwi.shape,
        }

        sample = {
            "dwi": dwi,
            "adc": adc,
            "flair": flair,
            "mask": mask,
            "metadata": metadata,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
