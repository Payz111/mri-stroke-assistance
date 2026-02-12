"""Training callbacks for checkpointing and early stopping.

Provides pluggable callback classes that the :class:`Trainer` invokes at
the end of each validation epoch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn


class CheckpointCallback:
    """Save model weights when the monitored metric improves.

    Parameters
    ----------
    save_dir:
        Directory where checkpoint files are written.
    monitor:
        Name of the metric to track (e.g. ``"val_dice"``).
    mode:
        ``"min"`` or ``"max"`` — whether lower or higher is better.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_dice",
        mode: str = "max",
    ) -> None:
        raise NotImplementedError()

    def __call__(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: nn.Module,
    ) -> None:
        """Evaluate whether to save a checkpoint.

        Parameters
        ----------
        epoch:
            Current epoch number.
        metrics:
            Validation metrics dictionary.
        model:
            The model whose state_dict should be saved.
        """
        raise NotImplementedError()


class EarlyStoppingCallback:
    """Stop training when the monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs to wait after last improvement.
    monitor:
        Name of the metric to track.
    mode:
        ``"min"`` or ``"max"``.
    min_delta:
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_dice",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        raise NotImplementedError()

    def __call__(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Check whether training should be stopped.

        Parameters
        ----------
        epoch:
            Current epoch number.
        metrics:
            Validation metrics dictionary.

        Returns
        -------
        bool
            ``True`` if training should stop, ``False`` otherwise.
        """
        raise NotImplementedError()
