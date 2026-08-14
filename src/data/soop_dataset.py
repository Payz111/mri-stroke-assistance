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
            anat/sub-{ID}_FLAIR.nii[.gz]
            dwi/sub-{ID}_rec-TRACE_dwi.nii[.gz]
            dwi/sub-{ID}_rec-ADC_dwi.nii[.gz]
          derivatives/lesion_masks/
            sub-{ID}/dwi/sub-{ID}_space-TRACE_desc-lesion_mask.nii[.gz]

    Distributions of this dataset differ in ways that are invisible until
    nothing loads, so path resolution tolerates all of them:

    * compressed ``.nii.gz`` or plain ``.nii``
    * a file wrapped in a directory of the same name, which some mirrors
      produce: ``anat/sub-1_FLAIR.nii/sub-1_FLAIR.nii``

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

    @staticmethod
    def _resolve(directory: Path, stem: str) -> Path | None:
        """Locate the NIfTI named *stem* in *directory*, whatever the layout.

        Accepts ``.nii`` and ``.nii.gz``, and the case where the archive wrapped
        the file inside a directory carrying the same name.
        """
        if not directory.is_dir():
            return None

        for suffix in (".nii.gz", ".nii"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
            if candidate.is_dir():
                inner = sorted(p for p in candidate.glob("*.nii*") if p.is_file())
                if inner:
                    return inner[0]

        matches = sorted(p for p in directory.glob(f"{stem}.nii*") if p.is_file())
        return matches[0] if matches else None

    def _discover_subjects(self) -> list[str]:
        """Find all subjects that have complete data (DWI + ADC + FLAIR + mask).

        Raises if *nothing* resolves. Silently returning an empty dataset is
        how a whole training run once completed on the wrong data without
        anyone noticing: the loader dropped 1323 subjects and said nothing.
        """
        subject_dirs = [d for d in sorted(self.data_root.iterdir()) if d.name.startswith("sub-")]

        subjects = []
        missing_counts: dict[str, int] = {}
        for sub_dir in subject_dirs:
            try:
                self._get_paths(sub_dir.name)
                subjects.append(sub_dir.name)
            except FileNotFoundError as exc:
                key = str(exc).split(": ", 1)[-1]
                missing_counts[key] = missing_counts.get(key, 0) + 1

        if subject_dirs and not subjects:
            raise FileNotFoundError(self._diagnose(subject_dirs, missing_counts))

        return subjects

    def _diagnose(self, subject_dirs: list[Path], missing_counts: dict[str, int]) -> str:
        """Explain why no subject resolved, listing what is actually on disk."""
        lines = [
            f"SOOPDataset found {len(subject_dirs)} sub-* directories under "
            f"{self.data_root} but none had a complete set of files "
            f"(require_mask={self.require_mask}).",
            "",
            "Missing-file tally:",
        ]
        for reason, count in sorted(missing_counts.items(), key=lambda kv: -kv[1])[:5]:
            lines.append(f"  {count:5d} x {reason}")

        sample = subject_dirs[0]
        lines += ["", f"What is actually inside {sample.name}:"]
        for sub in sorted(sample.rglob("*"))[:12]:
            kind = "dir " if sub.is_dir() else "file"
            lines.append(f"  {kind} {sub.relative_to(sample).as_posix()}")

        lines += [
            "",
            "Check that data_root points at the ds004889 root (the directory that",
            "directly contains the sub-* folders) and that the lesion masks live in",
            "derivatives/lesion_masks/. Pass require_mask=False to train without masks.",
        ]
        return "\n".join(lines)

    def _get_paths(self, subject_id: str) -> dict[str, Path]:
        """Resolve file paths for a SOOP subject."""
        sub_dir = self.data_root / subject_id

        dwi_path = self._resolve(sub_dir / "dwi", f"{subject_id}_rec-TRACE_dwi")
        adc_path = self._resolve(sub_dir / "dwi", f"{subject_id}_rec-ADC_dwi")
        flair_path = self._resolve(sub_dir / "anat", f"{subject_id}_FLAIR")

        # Lesion mask (in derivatives); prefer acute-only, fall back to combined
        mask_dir = self.data_root / "derivatives" / "lesion_masks" / subject_id / "dwi"
        mask_path = self._resolve(
            mask_dir, f"{subject_id}_space-TRACE_desc-lesionAcute_mask"
        ) or self._resolve(mask_dir, f"{subject_id}_space-TRACE_desc-lesion_mask")

        missing = []
        if dwi_path is None:
            missing.append("DWI")
        if adc_path is None:
            missing.append("ADC")
        if flair_path is None:
            missing.append("FLAIR")
        if self.require_mask and mask_path is None:
            missing.append("mask")

        if missing:
            raise FileNotFoundError(f"Missing files for {subject_id}: {missing}")

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
        for col in ("acuteischaemicstroke", "acuiteischaemicstroke", "acute_ischaemic_stroke"):
            val = info.get(col, "").strip().lower()
            if val == "yes":
                stroke_ids.append(pid)
                break
    return sorted(stroke_ids)
