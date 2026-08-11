"""Training callbacks for checkpointing and early stopping."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """Save model weights when the monitored metric improves."""

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_dice",
        mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = -1

    def __call__(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: nn.Module,
    ) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return

        improved = (value > self.best_value) if self.mode == "max" else (value < self.best_value)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            path = self.save_dir / "best_model.pth"
            torch.save(model.state_dict(), path)
            logger.info(
                "Checkpoint saved: epoch=%d, %s=%.4f -> %s",
                epoch,
                self.monitor,
                value,
                path,
            )


class EarlyStoppingCallback:
    """Stop training when the monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_dice",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def __call__(self, epoch: int, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.monitor)
        if value is None:
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                "Early stopping at epoch %d: no improvement for %d epochs",
                epoch,
                self.patience,
            )
            return True

        return False
