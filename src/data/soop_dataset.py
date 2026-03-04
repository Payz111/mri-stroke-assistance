"""PyTorch Dataset for the SOOP stroke dataset (OpenNeuro ds004889)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class SOOPDataset(Dataset):
    """Dataset for SOOP (Stroke Outcome Optimization Project).

    BIDS structure::

        ds004889/
          participants.tsv
          sub-{ID}/
            anat/sub-{ID}_FLAIR.nii.gz
            dwi/sub-{ID}_rec-TRACE_dwi.nii.gz
            dwi/sub-{ID}_rec-ADC_dwi.nii.gz
          derivatives/lesion_masks/
            sub-{ID}/dwi/sub-{ID}_space-TRACE_desc-lesion_mask.nii.gz

    Each item is a dict with keys: dwi, adc, flair, mask, metadata
    (same interface as ISLES22Dataset).
    """

    def __init__(
        self,
        data_root: str | Path,
        subject_ids: list[str] | None = None,
        transform: Any | None = None,
        require_mask: bool = True,
    ) -> None:
        """Initialize SOOP dataset.

        Parameters
        ----------
        data_root:
            Path to the ds004889 root directory.
        subject_ids:
            Explicit list of subject IDs to use. If None, auto-discovers
            all subjects that have DWI + ADC + FLAIR + lesion mask.
        transform:
            MONAI transform pipeline to apply.
        require_mask:
            If True (default), only include subjects with lesion masks.
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.require_mask = require_mask

        if subject_ids is not None:
            self.subject_ids = subject_ids
        else:
            self.subject_ids = self._discover_subjects()

    def _discover_subjects(self) -> list[str]:
        """Find all subjects that have complete data (DWI + ADC + FLAIR + mask)."""
        subjects = []
        for sub_dir in sorted(self.data_root.iterdir()):
            if not sub_dir.name.startswith("sub-"):
                continue
            sub_id = sub_dir.name
            try:
                self._get_paths(sub_id)
                subjects.append(sub_id)
            except FileNotFoundError:
                continue
        return subjects

    def _get_paths(self, subject_id: str) -> dict[str, Path]:
        """Resolve file paths for a SOOP subject."""
        sub_dir = self.data_root / subject_id

        # DWI (TRACE)
        dwi_path = sub_dir / "dwi" / f"{subject_id}_rec-TRACE_dwi.nii.gz"

        # ADC
        adc_path = sub_dir / "dwi" / f"{subject_id}_rec-ADC_dwi.nii.gz"

        # FLAIR
        flair_path = sub_dir / "anat" / f"{subject_id}_FLAIR.nii.gz"

        # Lesion mask (in derivatives)
        mask_dir = self.data_root / "derivatives" / "lesion_masks" / subject_id / "dwi"
        # Prefer acute-only mask, fall back to combined
        mask_acute = mask_dir / f"{subject_id}_space-TRACE_desc-lesionAcute_mask.nii.gz"
        mask_combined = mask_dir / f"{subject_id}_space-TRACE_desc-lesion_mask.nii.gz"

        if mask_acute.exists():
            mask_path = mask_acute
        elif mask_combined.exists():
            mask_path = mask_combined
        else:
            mask_path = None

        # Check required files
        missing = []
        if not dwi_path.exists():
            missing.append("DWI")
        if not adc_path.exists():
            missing.append("ADC")
        if not flair_path.exists():
            missing.append("FLAIR")
        if self.require_mask and mask_path is None:
            missing.append("mask")

        if missing:
            raise FileNotFoundError(
                f"Missing files for {subject_id}: {missing}"
            )

        result = {"dwi": dwi_path, "adc": adc_path, "flair": flair_path}
        if mask_path is not None:
            result["mask"] = mask_path
        return result

    @staticmethod
    def _to_3d(vol: np.ndarray) -> np.ndarray:
        """Ensure volume is 3D. If 4D, take the first volume."""
        if vol.ndim == 4:
            vol = vol[..., 0]
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D or 4D volume, got {vol.ndim}D")
        return vol

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Try loading this subject; on corrupt files, try next subject
        for attempt in range(min(10, len(self.subject_ids))):
            idx = (index + attempt) % len(self.subject_ids)
            subject_id = self.subject_ids[idx]
            try:
                return self._load_subject(subject_id)
            except (EOFError, OSError, ValueError) as e:
                import logging
                logging.warning("Skipping %s: %s", subject_id, e)
                continue
        raise RuntimeError(f"Could not load any subject starting from index {index}")

    @staticmethod
    def _reorient_to_canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
        """Reorient NIfTI image to canonical (RAS+) orientation.

        SOOP FLAIR can be coronal while DWI is axial. Without reorientation,
        the voxel arrays have incompatible axis ordering.
        """
        return nib.as_closest_canonical(img)

    def _load_subject(self, subject_id: str) -> dict[str, Any]:
        paths = self._get_paths(subject_id)

        # Reorient all modalities to canonical (RAS+) so axes are consistent
        dwi_img = self._reorient_to_canonical(nib.load(paths["dwi"]))
        adc_img = self._reorient_to_canonical(nib.load(paths["adc"]))
        flair_img = self._reorient_to_canonical(nib.load(paths["flair"]))

        dwi = self._to_3d(dwi_img.get_fdata(dtype=np.float32))
        adc = self._to_3d(adc_img.get_fdata(dtype=np.float32))
        flair = self._to_3d(flair_img.get_fdata(dtype=np.float32))

        if "mask" in paths:
            mask_img = self._reorient_to_canonical(nib.load(paths["mask"]))
            mask = self._to_3d(mask_img.get_fdata(dtype=np.float32))
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros_like(dwi, dtype=np.float32)

        metadata = {
            "subject_id": subject_id,
            "dataset": "soop",
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


def load_soop_participants(data_root: str | Path) -> dict[str, dict]:
    """Load participants.tsv and return dict keyed by participant_id."""
    tsv_path = Path(data_root) / "participants.tsv"
    participants = {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["participant_id"]
            participants[pid] = row
    return participants


def get_stroke_subject_ids(data_root: str | Path) -> list[str]:
    """Get subject IDs with confirmed acute ischemic stroke from participants.tsv."""
    participants = load_soop_participants(data_root)
    stroke_ids = []
    for pid, info in participants.items():
        # Try multiple column name variants
        for col in ("acuteischaemicstroke", "acuiteischaemicstroke",
                     "acute_ischaemic_stroke"):
            val = info.get(col, "").strip().lower()
            if val == "yes":
                stroke_ids.append(pid)
                break
    return sorted(stroke_ids)
