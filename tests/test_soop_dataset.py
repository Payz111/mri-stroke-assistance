"""Tests for SOOP path resolution and the loud-failure contract.

Background: a distribution of ds004889 stored plain `.nii` files, and wrapped
FLAIR in a directory of the same name. The loader expected `.nii.gz` only, so
`_get_paths` raised for all 1323 subjects and `_discover_subjects` swallowed
every exception -- a whole 13-hour training run completed on ISLES alone and
reported success. Both halves of that failure are covered here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.soop_dataset import SOOPDataset

SUBJECTS = ("sub-1", "sub-2")


def _write_nifti(path, shape=(8, 8, 4)):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), str(path))


def build_soop(root, suffix=".nii.gz", nested_flair=False, with_masks=True):
    """Create a miniature ds004889 tree in one of the real-world layouts."""
    for sid in SUBJECTS:
        _write_nifti(root / sid / "dwi" / f"{sid}_rec-TRACE_dwi{suffix}")
        _write_nifti(root / sid / "dwi" / f"{sid}_rec-ADC_dwi{suffix}")

        if nested_flair:
            # Some mirrors wrap the file in a directory carrying its own name
            _write_nifti(root / sid / "anat" / f"{sid}_FLAIR.nii" / f"{sid}_FLAIR.nii")
        else:
            _write_nifti(root / sid / "anat" / f"{sid}_FLAIR{suffix}")

        if with_masks:
            _write_nifti(
                root
                / "derivatives"
                / "lesion_masks"
                / sid
                / "dwi"
                / f"{sid}_space-TRACE_desc-lesion_mask{suffix}"
            )
    return root


class TestLayoutVariants:
    def test_compressed_layout(self, tmp_path):
        root = build_soop(tmp_path / "gz", suffix=".nii.gz")
        assert len(SOOPDataset(root).subject_ids) == 2

    def test_uncompressed_nii_layout(self, tmp_path):
        """The layout that silently produced an empty dataset."""
        root = build_soop(tmp_path / "plain", suffix=".nii")
        assert len(SOOPDataset(root).subject_ids) == 2

    def test_flair_wrapped_in_a_directory(self, tmp_path):
        root = build_soop(tmp_path / "nested", suffix=".nii", nested_flair=True)

        ds = SOOPDataset(root)

        assert len(ds.subject_ids) == 2
        flair = ds._get_paths("sub-1")["flair"]
        assert flair.is_file()
        assert flair.name.endswith(".nii")

    def test_acute_mask_is_preferred_over_combined(self, tmp_path):
        root = build_soop(tmp_path / "masks", suffix=".nii")
        mask_dir = root / "derivatives" / "lesion_masks" / "sub-1" / "dwi"
        _write_nifti(mask_dir / "sub-1_space-TRACE_desc-lesionAcute_mask.nii")

        assert "Acute" in SOOPDataset(root)._get_paths("sub-1")["mask"].name

    def test_require_mask_false_accepts_subjects_without_masks(self, tmp_path):
        root = build_soop(tmp_path / "nomask", suffix=".nii", with_masks=False)

        assert len(SOOPDataset(root, require_mask=False).subject_ids) == 2


class TestLoudFailure:
    def test_raises_instead_of_returning_an_empty_dataset(self, tmp_path):
        """The bug that cost a training run: 1323 subjects dropped in silence."""
        root = tmp_path / "broken"
        for sid in SUBJECTS:
            _write_nifti(root / sid / "dwi" / f"{sid}_unexpected_name.nii")

        with pytest.raises(FileNotFoundError) as excinfo:
            SOOPDataset(root)

        message = str(excinfo.value)
        assert "2 sub-* directories" in message
        assert "none had a complete set" in message

    def test_error_message_lists_what_is_actually_on_disk(self, tmp_path):
        """The diagnosis has to be actionable without opening a shell."""
        root = tmp_path / "broken2"
        _write_nifti(root / "sub-1" / "dwi" / "sub-1_wrong.nii")

        with pytest.raises(FileNotFoundError) as excinfo:
            SOOPDataset(root)

        message = str(excinfo.value)
        assert "sub-1_wrong.nii" in message
        assert "require_mask" in message

    def test_missing_masks_are_reported_as_the_reason(self, tmp_path):
        root = build_soop(tmp_path / "maskless", suffix=".nii", with_masks=False)

        with pytest.raises(FileNotFoundError) as excinfo:
            SOOPDataset(root, require_mask=True)

        assert "mask" in str(excinfo.value)

    def test_empty_root_is_not_an_error(self, tmp_path):
        """No sub-* directories at all is a different problem; stay quiet."""
        root = tmp_path / "empty"
        root.mkdir()

        assert SOOPDataset(root).subject_ids == []

    def test_explicit_subject_ids_skip_discovery(self, tmp_path):
        root = build_soop(tmp_path / "explicit", suffix=".nii")

        ds = SOOPDataset(root, subject_ids=["sub-1"])

        assert ds.subject_ids == ["sub-1"]
