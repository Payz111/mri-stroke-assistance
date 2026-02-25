"""PyTorch Dataset for the ISLES'22 ischemic-stroke segmentation challenge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class ISLES22Dataset(Dataset):
    """Dataset class that serves ISLES'22 subjects.

    Each item is a dictionary with the keys:
        - ``dwi``      : DWI volume (C, D, H, W) float32
        - ``adc``      : ADC volume (C, D, H, W) float32
        - ``flair``    : FLAIR volume (C, D, H, W) float32
        - ``mask``     : Ground-truth lesion mask (C, D, H, W) float32
        - ``metadata`` : dict with subject_id, spacing, affine, shape
    """

    def __init__(
        self,
        data_root: str | Path,
        derivatives_root: str | Path,
        split_file: str | Path,
        split: str = "train",
        transform: Any | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.derivatives_root = Path(derivatives_root)
        self.transform = transform

        # Load split
        with open(split_file) as f:
            split_data = json.load(f)

        self.subject_ids = split_data[split]

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _get_paths(self, subject_id: str) -> dict[str, Path]:
        """Resolve file paths for a subject.

        Supports both compressed (.nii.gz) and uncompressed (.nii) NIfTI files,
        and both flat and nested directory layouts (e.g. Kaggle vs local).
        """
        sub_path = self.data_root / subject_id / "ses-0001"
        deriv_path = self.derivatives_root / subject_id / "ses-0001"

        # Search for .nii.gz first, fall back to .nii (also search subdirs)
        dwi_files = (list(sub_path.glob("dwi/*dwi.nii.gz"))
                     or list(sub_path.glob("dwi/**/*dwi.nii")))
        adc_files = (list(sub_path.glob("dwi/*adc.nii.gz"))
                     or list(sub_path.glob("dwi/**/*adc.nii")))
        flair_files = (list(sub_path.glob("anat/*FLAIR.nii.gz"))
                       or list(sub_path.glob("anat/*FLAIR.nii")))
        mask_files = (list(deriv_path.glob("*msk.nii.gz"))
                      or list(deriv_path.glob("*msk.nii")))

        if not (dwi_files and adc_files and flair_files and mask_files):
            missing = []
            if not dwi_files:
                missing.append("DWI")
            if not adc_files:
                missing.append("ADC")
            if not flair_files:
                missing.append("FLAIR")
            if not mask_files:
                missing.append("mask")
            raise FileNotFoundError(
                f"Missing files for {subject_id}: {missing}"
            )

        return {
            "dwi": dwi_files[0],
            "adc": adc_files[0],
            "flair": flair_files[0],
            "mask": mask_files[0],
        }

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
