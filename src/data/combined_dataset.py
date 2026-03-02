"""Combined dataset for training on multiple stroke datasets (ISLES + SOOP)."""

from __future__ import annotations

from torch.utils.data import ConcatDataset, Dataset


class CombinedStrokeDataset(ConcatDataset):
    """Concatenates multiple stroke datasets into a single training set.

    Usage::

        combined = CombinedStrokeDataset([isles_ds, soop_ds])
        loader = DataLoader(combined, batch_size=4, shuffle=True)
    """

    def __init__(self, datasets: list[Dataset]) -> None:
        super().__init__(datasets)
        self.dataset_names = []
        self.dataset_sizes = []
        for ds in datasets:
            name = type(ds).__name__
            size = len(ds)
            self.dataset_names.append(name)
            self.dataset_sizes.append(size)

    def summary(self) -> str:
        parts = []
        for name, size in zip(self.dataset_names, self.dataset_sizes):
            parts.append(f"{name}: {size}")
        total = sum(self.dataset_sizes)
        return f"CombinedStrokeDataset({', '.join(parts)}, total={total})"
