"""Tests for the packed, memory-mapped volume cache.

The pack is built once and reused across all five folds, so the failure that
matters most is a silent mismatch: a split whose subjects are only partly in
the cache would train on the wrong data and still look healthy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.data.volume_cache import (
    INDEX_FILE,
    PackedVolumeDataset,
    build_pack,
    find_cache,
)

SHAPE = (40, 40, 20)
SPATIAL = (32, 32, 16)


class RawSubjects(Dataset):
    def __init__(self, ids, broken=()):
        self.subject_ids = list(ids)
        self.broken = set(broken)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        sid = self.subject_ids[index]
        if sid in self.broken:
            raise OSError(f"damaged file for {sid}")
        rng = np.random.default_rng(abs(hash(sid)) % 10_000)
        volume = rng.uniform(50, 500, SHAPE).astype(np.float32)
        mask = np.zeros(SHAPE, dtype=np.float32)
        mask[10:20, 10:20, 5:12] = 1.0
        return {
            "dwi": volume,
            "adc": volume * 1.7,
            "flair": volume * 0.8,
            "mask": mask,
            "metadata": {"subject_id": sid, "spacing": (2.0, 2.0, 2.0)},
        }


@pytest.fixture
def pack(tmp_path):
    isles = RawSubjects(["sub-strokecase0001", "sub-strokecase0002"])
    soop = RawSubjects(["sub-1", "sub-2", "sub-3"])
    build_pack([isles, soop], tmp_path, spatial_size=SPATIAL, log_every=0)
    return tmp_path


class TestBuildPack:
    def test_writes_three_files(self, pack):
        names = {p.name for p in pack.iterdir()}
        assert {"images.npy", "labels.npy", INDEX_FILE} <= names

    def test_index_records_every_subject(self, pack):
        index = json.loads((pack / INDEX_FILE).read_text(encoding="utf-8"))

        assert index["n_subjects"] == 5
        assert set(index["subjects"]) == {
            "sub-strokecase0001",
            "sub-strokecase0002",
            "sub-1",
            "sub-2",
            "sub-3",
        }
        assert index["spatial_size"] == list(SPATIAL)

    def test_stays_well_under_the_kaggle_file_limit(self, pack):
        """1321 per-subject files would exceed Kaggle's 500-file output cap."""
        assert len(list(pack.iterdir())) < 10

    def test_rejects_colliding_subject_ids(self, tmp_path):
        a = RawSubjects(["sub-1", "sub-2"])
        b = RawSubjects(["sub-2", "sub-3"])

        with pytest.raises(ValueError, match="collide"):
            build_pack([a, b], tmp_path, spatial_size=SPATIAL, log_every=0)

    def test_damaged_subject_is_skipped_not_fatal(self, tmp_path):
        source = RawSubjects(["sub-1", "sub-502", "sub-3"], broken={"sub-502"})

        index = build_pack([source], tmp_path, spatial_size=SPATIAL, log_every=0)

        assert index["n_subjects"] == 2
        assert index["skipped"] == ["sub-502"]
        assert "sub-502" not in index["subjects"]

    def test_arrays_are_trimmed_after_skips(self, tmp_path):
        """Rows left over from skipped subjects must not linger as zeros."""
        source = RawSubjects(["sub-1", "sub-502", "sub-3"], broken={"sub-502"})
        build_pack([source], tmp_path, spatial_size=SPATIAL, log_every=0)

        images = np.load(tmp_path / "images.npy", mmap_mode="r")

        assert images.shape[0] == 2


class TestPackedVolumeDataset:
    def test_returns_tensors_at_the_cached_resolution(self, pack):
        ds = PackedVolumeDataset(pack)

        sample = ds[0]

        assert sample["image"].shape == (3, *SPATIAL)
        assert sample["label"].shape == (1, *SPATIAL)
        assert isinstance(sample["image"], torch.Tensor)

    def test_subject_selection_defines_the_split(self, pack):
        """One pack serves every fold; the split is just a list of ids."""
        train = PackedVolumeDataset(pack, ["sub-1", "sub-2", "sub-strokecase0001"])
        val = PackedVolumeDataset(pack, ["sub-strokecase0002"])

        assert len(train) == 3
        assert len(val) == 1
        assert val[0]["subject_id"] == "sub-strokecase0002"

    def test_order_follows_the_requested_ids(self, pack):
        ds = PackedVolumeDataset(pack, ["sub-3", "sub-1"])

        assert [ds[i]["subject_id"] for i in range(len(ds))] == ["sub-3", "sub-1"]

    def test_unknown_subject_raises_instead_of_being_dropped(self, pack):
        """Silently shrinking the split is how a run trains on the wrong data."""
        with pytest.raises(KeyError, match="not in the cache"):
            PackedVolumeDataset(pack, ["sub-1", "sub-does-not-exist"])

    def test_missing_index_explains_itself(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_volume_cache"):
            PackedVolumeDataset(tmp_path)

    def test_content_is_stable_without_augmentation(self, pack):
        ds = PackedVolumeDataset(pack, augment=False)

        first = ds[0]["image"].clone()

        assert torch.allclose(first, ds[0]["image"])

    def test_augmentation_varies_between_reads(self, pack):
        ds = PackedVolumeDataset(pack, augment=True)

        draws = [ds[0]["image"].clone() for _ in range(12)]

        assert any(not torch.allclose(draws[0], other) for other in draws[1:])

    def test_labels_stay_binary(self, pack):
        ds = PackedVolumeDataset(pack, augment=True)

        for _ in range(5):
            assert set(torch.unique(ds[0]["label"]).tolist()) <= {0.0, 1.0}

    def test_reads_are_lazy_so_workers_get_their_own_handles(self, pack):
        ds = PackedVolumeDataset(pack)

        assert ds._images is None  # nothing opened until first access
        ds[0]
        assert ds._images is not None


class TestFindCache:
    def test_finds_the_pack_directly(self, pack):
        assert find_cache([pack]) == pack

    def test_finds_a_nested_pack(self, tmp_path):
        """Kaggle mounts the cache several levels below /kaggle/input."""
        mount = tmp_path / "input" / "datasets" / "owner" / "stroke-cache"
        mount.mkdir(parents=True)
        build_pack([RawSubjects(["sub-1"])], mount, spatial_size=SPATIAL, log_every=0)

        assert find_cache([tmp_path / "input"]) == mount

    def test_returns_none_when_absent(self, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        assert find_cache([empty]) is None
