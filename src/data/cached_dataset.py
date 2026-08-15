"""Cache the deterministic half of preprocessing to disk.

Profiling one ISLES subject showed where epoch time goes:

    read NIfTI   502 ms   (FLAIR alone is 139 MB / 34.8 M voxels)
    resample      97 ms
    normalise     14 ms
    stack          7 ms

All of it produces the same 128x128x80 volume every epoch, and all of it was
repeated every epoch. Cached, the same sample loads in 22.6 ms -- 25.7x faster,
which removes roughly 370 s from a 1000 s epoch on the combined dataset.

Augmentation is deliberately *not* cached: it must redraw every epoch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.data.transforms import get_augment_transforms, get_deterministic_transforms

CACHE_VERSION = 1


class CachedDataset(Dataset):
    """Wrap a dataset so deterministic preprocessing happens once per subject.

    The wrapped dataset must yield the raw sample dict (``dwi``, ``adc``,
    ``flair``, ``mask``, ``metadata``) with **no** transform applied -- this
    class owns the transform pipeline.

    Parameters
    ----------
    source:
        Dataset returning raw, untransformed samples.
    cache_dir:
        Where the preprocessed arrays live. Roughly 9.2 MB per subject, so
        about 12 GB for the 1321-subject combined set. On Kaggle put this on
        /kaggle/temp, not /kaggle/working.
    augment:
        Apply random augmentation after loading. True for training.
    spatial_size:
        Overrides the model resolution; changes the cache key.
    """

    def __init__(
        self,
        source: Dataset,
        cache_dir: str | Path,
        augment: bool = False,
        spatial_size: tuple[int, int, int] | None = None,
    ) -> None:
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.augment = augment

        config = {"spatial_size": list(spatial_size)} if spatial_size else None
        self._deterministic = get_deterministic_transforms(config)
        self._augment = get_augment_transforms() if augment else None
        self._spatial_size = spatial_size

    def __len__(self) -> int:
        return len(self.source)

    def _cache_path(self, index: int) -> Path:
        """A key that changes whenever the cached content would."""
        subject = self._subject_id(index)
        parts = f"v{CACHE_VERSION}|{subject}|{self._spatial_size}"
        digest = hashlib.sha1(parts.encode()).hexdigest()[:12]
        return self.cache_dir / f"{subject}_{digest}.npz"

    def _subject_id(self, index: int) -> str:
        ids = getattr(self.source, "subject_ids", None)
        if ids is not None and index < len(ids):
            return str(ids[index])
        return f"idx{index:06d}"

    def _build(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.source[index]
        processed = self._deterministic(sample)
        image = np.asarray(processed["image"], dtype=np.float16)
        label = (np.asarray(processed["label"]) > 0).astype(np.uint8)
        return image, label

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self._cache_path(index)

        if path.is_file():
            try:
                with np.load(path) as blob:
                    image = blob["image"]
                    label = blob["label"]
            except (OSError, ValueError, EOFError):
                # A truncated cache entry (interrupted write) must not kill the
                # run; rebuild it and carry on.
                path.unlink(missing_ok=True)
                image, label = self._build(index)
                self._write(path, image, label)
        else:
            image, label = self._build(index)
            self._write(path, image, label)

        sample: dict[str, Any] = {
            "image": np.asarray(image, dtype=np.float32),
            "label": np.asarray(label, dtype=np.float32),
        }
        if self._augment is not None:
            sample = self._augment(sample)

        import torch

        return {
            "image": torch.as_tensor(np.ascontiguousarray(sample["image"]), dtype=torch.float32),
            "label": torch.as_tensor(np.ascontiguousarray(sample["label"]), dtype=torch.float32),
        }

    @staticmethod
    def _write(path: Path, image: np.ndarray, label: np.ndarray) -> None:
        """Write atomically: a half-written file must never be read as valid."""
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, image=image, label=label)
        tmp.replace(path)

    def warm(self, log_every: int = 100) -> int:
        """Build every cache entry up front, reporting progress.

        Doing this once before training keeps the first epoch from being the
        slow one and surfaces unreadable subjects immediately.
        """
        built = 0
        for index in range(len(self)):
            path = self._cache_path(index)
            if not path.is_file():
                image, label = self._build(index)
                self._write(path, image, label)
                built += 1
            if log_every and (index + 1) % log_every == 0:
                print(f"  cached {index + 1}/{len(self)}")
        return built
