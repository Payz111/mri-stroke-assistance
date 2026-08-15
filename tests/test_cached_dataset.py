"""Tests for the preprocessing cache.

The cache must be transparent: same shapes, same content, augmentation still
random per epoch. Getting that last part wrong would freeze one augmentation
per subject forever and quietly weaken training.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.data.cached_dataset import CachedDataset

SHAPE = (40, 40, 20)


class RawSubjects(Dataset):
    """Stand-in for ISLES22Dataset/SOOPDataset yielding untransformed samples."""

    def __init__(self, n=3):
        self.subject_ids = [f"sub-{i}" for i in range(n)]
        self.reads = 0

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        self.reads += 1
        rng = np.random.default_rng(index)
        volume = rng.uniform(50, 500, SHAPE).astype(np.float32)
        mask = np.zeros(SHAPE, dtype=np.float32)
        mask[10:20, 10:20, 5:12] = 1.0
        return {
            "dwi": volume,
            "adc": volume * 1.7,
            "flair": volume * 0.8,
            "mask": mask,
            "metadata": {"subject_id": self.subject_ids[index], "spacing": (2.0, 2.0, 2.0)},
        }


@pytest.fixture
def source():
    return RawSubjects()


class TestCacheBehaviour:
    def test_returns_tensors_at_model_resolution(self, tmp_path, source):
        ds = CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))

        sample = ds[0]

        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, 32, 32, 16)
        assert sample["label"].shape == (1, 32, 32, 16)

    def test_source_is_read_once_per_subject(self, tmp_path, source):
        """The whole point: the second epoch must not touch the NIfTI files."""
        ds = CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))

        for _ in range(3):
            for i in range(len(ds)):
                ds[i]

        assert source.reads == len(ds)

    def test_cached_content_matches_the_first_computation(self, tmp_path, source):
        ds = CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))

        first = ds[1]["image"].clone()
        second = ds[1]["image"]

        assert torch.allclose(first, second)

    def test_cache_files_are_written(self, tmp_path, source):
        ds = CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))
        ds[0]

        assert list(tmp_path.glob("sub-0_*.npz"))

    def test_no_temporary_files_are_left_behind(self, tmp_path, source):
        ds = CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))
        ds[0]

        assert not list(tmp_path.glob("*.tmp.npz"))

    def test_warm_builds_every_entry(self, tmp_path, source):
        ds = CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))

        built = ds.warm(log_every=0)

        assert built == len(ds)
        assert len(list(tmp_path.glob("*.npz"))) == len(ds)
        assert ds.warm(log_every=0) == 0  # already warm

    def test_spatial_size_changes_the_cache_key(self, tmp_path, source):
        """A different model resolution must not reuse the old arrays."""
        CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))[0]
        CachedDataset(source, tmp_path, spatial_size=(24, 24, 8))[0]

        assert len(list(tmp_path.glob("sub-0_*.npz"))) == 2

    def test_corrupt_entry_is_rebuilt(self, tmp_path, source):
        """An interrupted write must not poison the run."""
        ds = CachedDataset(source, tmp_path, spatial_size=(32, 32, 16))
        ds[0]

        cache_file = next(tmp_path.glob("sub-0_*.npz"))
        cache_file.write_bytes(b"not an npz")

        sample = ds[0]

        assert sample["image"].shape == (3, 32, 32, 16)


class TestAugmentation:
    def test_augmentation_is_not_frozen_by_the_cache(self, tmp_path, source):
        """Cached augmentation would show every subject the same way forever."""
        ds = CachedDataset(source, tmp_path, augment=True, spatial_size=(32, 32, 16))

        draws = [ds[0]["image"].clone() for _ in range(12)]

        assert any(not torch.allclose(draws[0], other) for other in draws[1:])

    def test_without_augment_the_sample_is_deterministic(self, tmp_path, source):
        ds = CachedDataset(source, tmp_path, augment=False, spatial_size=(32, 32, 16))

        draws = [ds[0]["image"].clone() for _ in range(5)]

        assert all(torch.allclose(draws[0], other) for other in draws[1:])

    def test_label_stays_binary_through_augmentation(self, tmp_path, source):
        ds = CachedDataset(source, tmp_path, augment=True, spatial_size=(32, 32, 16))

        for _ in range(5):
            values = set(torch.unique(ds[0]["label"]).tolist())
            assert values.issubset({0.0, 1.0})
