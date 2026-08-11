"""Smoke test: load 1 case from ISLES'22 dataset, apply transforms, verify output."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.data.isles22_dataset import ISLES22Dataset
from src.data.transforms import get_val_transforms


def main() -> None:
    data_root = project_root / "data" / "raw" / "isles22" / "ISLES-2022"
    derivatives_root = data_root / "derivatives"
    split_file = project_root / "data" / "splits" / "fold_0.json"

    print("=== Smoke Test: ISLES22Dataset ===\n")

    # 1. Load without transforms
    print("1) Loading dataset WITHOUT transforms...")
    ds_raw = ISLES22Dataset(
        data_root=data_root,
        derivatives_root=derivatives_root,
        split_file=split_file,
        split="train",
    )
    print(f"   Dataset size: {len(ds_raw)} subjects")

    sample = ds_raw[0]
    sid = sample["metadata"]["subject_id"]
    print(f"   Subject: {sid}")
    for key in ["dwi", "adc", "flair", "mask"]:
        v = sample[key]
        print(
            f"   {key:6s}: shape={v.shape}, dtype={v.dtype}, "
            f"min={v.min():.2f}, max={v.max():.2f}, mean={v.mean():.2f}"
        )
    print(f"   spacing: {sample['metadata']['spacing']}")
    print(f"   original shape: {sample['metadata']['shape']}")

    # 2. Load with transforms
    print("\n2) Loading dataset WITH val transforms...")
    ds_val = ISLES22Dataset(
        data_root=data_root,
        derivatives_root=derivatives_root,
        split_file=split_file,
        split="train",
        transform=get_val_transforms(),
    )

    sample_t = ds_val[0]
    print(f"   Subject: {sid}")
    for key in ["image", "label"]:
        v = sample_t[key]
        print(
            f"   {key:6s}: shape={v.shape}, dtype={v.dtype}, "
            f"min={v.min():.4f}, max={v.max():.4f}"
        )

    # 3. Verify shapes
    img = sample_t["image"]
    lbl = sample_t["label"]
    assert img.ndim == 4, f"Expected 4D image, got {img.ndim}D"
    assert img.shape[0] == 3, f"Expected 3 channels, got {img.shape[0]}"
    assert lbl.ndim == 4, f"Expected 4D label, got {lbl.ndim}D"
    assert lbl.shape[0] == 1, f"Expected 1 channel label, got {lbl.shape[0]}"
    assert (
        img.shape[1:] == lbl.shape[1:]
    ), f"Spatial mismatch: image {img.shape[1:]} vs label {lbl.shape[1:]}"

    # 4. Check mask is binary
    unique_vals = np.unique(lbl)
    assert all(v in [0.0, 1.0] for v in unique_vals), f"Mask not binary: {unique_vals}"
    print(f"\n   Mask unique values: {unique_vals}")
    print(f"   Lesion voxels: {int(lbl.sum())}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
