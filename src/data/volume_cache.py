"""A single-file, memory-mapped cache of preprocessed volumes.

Why a pack rather than one file per subject:

* Kaggle allows at most 500 files in a notebook's output, and the combined set
  has 1321 subjects.
* A Kaggle Dataset is mounted read-only at /kaggle/input. A memmap is read
  straight from that mount, so attaching the cache costs nothing at session
  start -- no 12 GB unpack before training can begin.
* The cache is fold-independent: folds only change which subjects are train and
  which are validation, so one pack serves all five.

Layout::

    images.npy   float16 (N, 3, D, H, W)   z-scored DWI/ADC/FLAIR
    labels.npy   uint8   (N, 1, D, H, W)   binary lesion mask
    index.json   subject_id -> row, plus spatial size and version

Augmentation is never stored here; it must redraw every epoch.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import get_augment_transforms, get_deterministic_transforms

PACK_VERSION = 1
IMAGES_FILE = "images.npy"
LABELS_FILE = "labels.npy"
INDEX_FILE = "index.json"


def _open_memmap(path: Path, dtype, shape, mode: str) -> np.memmap:
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


def build_pack(
    sources: Iterable[Dataset],
    out_dir: str | Path,
    spatial_size: tuple[int, int, int] = (128, 128, 80),
    log_every: int = 50,
) -> dict[str, Any]:
    """Preprocess every subject once and write the pack.

    Parameters
    ----------
    sources:
        Datasets yielding *raw* samples (no transform applied) and exposing
        ``subject_ids``.
    out_dir:
        Destination directory.
    spatial_size:
        Model resolution; recorded in the index so a mismatch is caught.
    log_every:
        Progress cadence, 0 to silence.

    Returns
    -------
    dict
        The index that was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = list(sources)
    subject_ids: list[str] = []
    for source in sources:
        ids = getattr(source, "subject_ids", None)
        if ids is None:
            raise AttributeError(f"{type(source).__name__} has no subject_ids")
        subject_ids.extend(str(i) for i in ids)

    duplicates = {s for s in subject_ids if subject_ids.count(s) > 1}
    if duplicates:
        raise ValueError(f"Subject IDs collide across sources: {sorted(duplicates)[:5]}")

    total = len(subject_ids)
    depth, height, width = spatial_size
    images = _open_memmap(out_dir / IMAGES_FILE, np.float16, (total, 3, depth, height, width), "w+")
    labels = _open_memmap(out_dir / LABELS_FILE, np.uint8, (total, 1, depth, height, width), "w+")

    deterministic = get_deterministic_transforms({"spatial_size": list(spatial_size)})

    row = 0
    written: dict[str, int] = {}
    skipped: list[str] = []
    for source in sources:
        for local_index in range(len(source)):
            subject = str(source.subject_ids[local_index])
            try:
                processed = deterministic(source[local_index])
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the build
                skipped.append(subject)
                print(f"  SKIP {subject}: {type(exc).__name__}: {exc}")
                continue

            images[row] = np.asarray(processed["image"], dtype=np.float16)
            labels[row] = (np.asarray(processed["label"]) > 0).astype(np.uint8)
            written[subject] = row
            row += 1

            if log_every and row % log_every == 0:
                print(f"  packed {row}/{total}")

    images.flush()
    labels.flush()
    del images, labels

    # Trim the trailing rows left by skipped subjects.
    if row != total:
        _truncate(out_dir / IMAGES_FILE, np.float16, (row, 3, depth, height, width))
        _truncate(out_dir / LABELS_FILE, np.uint8, (row, 1, depth, height, width))

    index = {
        "version": PACK_VERSION,
        "spatial_size": list(spatial_size),
        "n_subjects": row,
        "subjects": written,
        "skipped": skipped,
    }
    (out_dir / INDEX_FILE).write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Pack written: {row} subjects, {len(skipped)} skipped -> {out_dir}")
    return index


def _truncate(path: Path, dtype, shape) -> None:
    """Rewrite an over-allocated memmap at its true length."""
    source = np.load(path, mmap_mode="r")
    target_path = path.with_suffix(".trimmed.npy")
    target = _open_memmap(target_path, dtype, shape, "w+")
    target[:] = source[: shape[0]]
    target.flush()
    del source, target
    target_path.replace(path)


class PackedVolumeDataset(Dataset):
    """Read preprocessed volumes from a pack, optionally augmenting.

    Parameters
    ----------
    cache_dir:
        Directory holding images.npy, labels.npy and index.json.
    subject_ids:
        Subjects to expose, in this order. None uses every subject in the pack.
        Unknown IDs are reported rather than silently dropped -- a split that
        half-matches the cache would train on the wrong data.
    augment:
        Apply random augmentation. True for training, False for validation.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        subject_ids: Sequence[str] | None = None,
        augment: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError(
                f"No cache index at {index_path}. Build it with scripts/build_volume_cache.py "
                "or attach the prebuilt cache dataset."
            )
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.spatial_size = tuple(self.index["spatial_size"])

        available = self.index["subjects"]
        if subject_ids is None:
            wanted = list(available)
        else:
            wanted = [str(s) for s in subject_ids]
            missing = [s for s in wanted if s not in available]
            if missing:
                raise KeyError(
                    f"{len(missing)} of {len(wanted)} requested subjects are not in the cache "
                    f"(e.g. {missing[:3]}). The pack may predate this split, or was built "
                    "from different sources."
                )

        self.subject_ids = wanted
        self.rows = [available[s] for s in wanted]
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None
        self._augment = get_augment_transforms() if augment else None

    def _arrays(self) -> tuple[np.memmap, np.memmap]:
        # Opened lazily so DataLoader worker processes each get their own handle.
        if self._images is None:
            self._images = np.load(self.cache_dir / IMAGES_FILE, mmap_mode="r")
            self._labels = np.load(self.cache_dir / LABELS_FILE, mmap_mode="r")
        return self._images, self._labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        images, labels = self._arrays()
        row = self.rows[index]

        sample: dict[str, Any] = {
            "image": np.asarray(images[row], dtype=np.float32),
            "label": np.asarray(labels[row], dtype=np.float32),
        }
        if self._augment is not None:
            sample = self._augment(sample)

        return {
            "image": torch.as_tensor(np.ascontiguousarray(sample["image"]), dtype=torch.float32),
            "label": torch.as_tensor(np.ascontiguousarray(sample["label"]), dtype=torch.float32),
            "subject_id": self.subject_ids[index],
        }


def find_cache(search_roots: Iterable[str | Path]) -> Path | None:
    """Locate a pack under any of *search_roots* (for Kaggle input mounts)."""
    for root in search_roots:
        root = Path(root)
        if (root / INDEX_FILE).is_file():
            return root
        if not root.is_dir():
            continue
        for candidate in root.rglob(INDEX_FILE):
            return candidate.parent
    return None
