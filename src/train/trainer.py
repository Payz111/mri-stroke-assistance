"""Training loop orchestration for stroke-lesion segmentation models."""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.callbacks import CheckpointCallback, EarlyStoppingCallback

logger = logging.getLogger(__name__)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """Compute Dice coefficient from sigmoid logits and binary target."""
    with torch.no_grad():
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        intersection = (pred_bin * target).sum()
        union = pred_bin.sum() + target.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


class Trainer:
    """High-level trainer that manages the train/val loop."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        scheduler: Any | None = None,
        callbacks: Sequence[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.callbacks = list(callbacks or [])
        self.config = config or {}
        self.history: list[dict[str, float]] = []

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_dice += compute_dice(logits, labels)
            n_batches += 1

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_dice": total_dice / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            total_dice += compute_dice(logits, labels)
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_dice": total_dice / max(n_batches, 1),
        }

    def fit(self, num_epochs: int) -> dict[str, Any]:
        """Execute the full training loop."""
        logger.info("Starting training for %d epochs on %s", num_epochs, self.device)
        best_val_dice = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            self.history.append(metrics)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d (%.1fs): train_loss=%.4f train_dice=%.4f | "
                "val_loss=%.4f val_dice=%.4f",
                epoch + 1, num_epochs, elapsed,
                metrics["train_loss"], metrics["train_dice"],
                metrics["val_loss"], metrics["val_dice"],
            )

            if metrics["val_dice"] > best_val_dice:
                best_val_dice = metrics["val_dice"]
                best_epoch = epoch

            # Run callbacks
            stop = False
            for cb in self.callbacks:
                if isinstance(cb, CheckpointCallback):
                    cb(epoch, metrics, self.model)
                elif isinstance(cb, EarlyStoppingCallback):
                    if cb(epoch, metrics):
                        stop = True

            if stop:
                break

        logger.info(
            "Training complete. Best val_dice=%.4f at epoch %d",
            best_val_dice, best_epoch + 1,
        )

        return {
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "history": self.history,
            "total_epochs": len(self.history),
        }
