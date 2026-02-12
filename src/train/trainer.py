"""Training loop orchestration for stroke-lesion segmentation models.

Encapsulates epoch-level training, validation, metric logging, and the
overall ``fit`` loop with callback support.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """High-level trainer that manages the train/val loop.

    Parameters
    ----------
    model:
        The segmentation model to train.
    optimizer:
        PyTorch optimiser instance.
    criterion:
        Loss function module.
    train_loader:
        DataLoader for the training split.
    val_loader:
        DataLoader for the validation split.
    device:
        Target device (``"cuda"`` or ``"cpu"``).
    callbacks:
        Optional sequence of callback objects (e.g.
        :class:`CheckpointCallback`, :class:`EarlyStoppingCallback`).
    config:
        Experiment configuration dictionary.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        callbacks: Sequence[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Parameters
        ----------
        epoch:
            Current epoch number (zero-based).

        Returns
        -------
        dict[str, float]
            Training metrics for this epoch (e.g. ``loss``, ``dice``).
        """
        raise NotImplementedError()

    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """Run one validation epoch.

        Parameters
        ----------
        epoch:
            Current epoch number (zero-based).

        Returns
        -------
        dict[str, float]
            Validation metrics for this epoch.
        """
        raise NotImplementedError()

    def fit(self, num_epochs: int) -> dict[str, Any]:
        """Execute the full training loop.

        Parameters
        ----------
        num_epochs:
            Total number of epochs to train.

        Returns
        -------
        dict[str, Any]
            Summary of the run including best metrics and checkpoint path.
        """
        raise NotImplementedError()
