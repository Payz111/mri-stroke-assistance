"""Training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --fold 0 --epochs 100 --batch-size 2
    python scripts/train.py --epochs 5 --device cpu   # smoke test
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Allow imports from src/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.isles22_dataset import ISLES22Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import create_loss, create_model
from src.train.callbacks import CheckpointCallback, EarlyStoppingCallback
from src.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D U-Net on ISLES'22")
    parser.add_argument("--fold", type=int, default=0, help="CV fold index (0-4)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    return parser.parse_args()


def load_config(config_path: Path | None = None) -> dict:
    """Load default config, optionally overridden by experiment config."""
    default_cfg_path = project_root / "configs" / "default.yaml"
    with open(default_cfg_path) as f:
        cfg = yaml.safe_load(f)

    if config_path is not None:
        with open(config_path) as f:
            override = yaml.safe_load(f)
        # Merge (shallow per top-level key)
        for key, val in override.items():
            if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(val)
            else:
                cfg[key] = val

    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)

    # Apply CLI overrides
    fold = args.fold
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    num_workers = args.num_workers

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logging.info("Config: fold=%d, epochs=%d, batch_size=%d, lr=%.1e, device=%s",
                 fold, epochs, batch_size, lr, device)

    # Paths
    data_root = project_root / cfg["paths"]["data_raw"] / "ISLES-2022"
    derivatives_root = data_root / "derivatives"
    split_file = project_root / cfg["paths"]["splits"] / f"fold_{fold}.json"
    output_dir = project_root / cfg["paths"]["outputs"] / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = ISLES22Dataset(
        data_root=data_root,
        derivatives_root=derivatives_root,
        split_file=split_file,
        split="train",
        transform=get_train_transforms(cfg.get("augmentation")),
    )
    val_ds = ISLES22Dataset(
        data_root=data_root,
        derivatives_root=derivatives_root,
        split_file=split_file,
        split="val",
        transform=get_val_transforms(),
    )

    logging.info("Train: %d subjects, Val: %d subjects", len(train_ds), len(val_ds))

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model
    model = create_model(cfg["model"])
    param_count = sum(p.numel() for p in model.parameters())
    logging.info("Model: %s, params: %s", cfg["model"]["name"], f"{param_count:,}")

    # Loss
    criterion = create_loss(cfg["loss"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7,
    )

    # Callbacks
    callbacks = [
        CheckpointCallback(save_dir=output_dir / "checkpoints", monitor="val_dice"),
        EarlyStoppingCallback(patience=15, monitor="val_dice"),
    ]

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scheduler=scheduler,
        callbacks=callbacks,
        config=cfg,
    )

    # Train
    result = trainer.fit(num_epochs=epochs)
    logging.info("Best val_dice=%.4f at epoch %d", result["best_val_dice"], result["best_epoch"] + 1)


if __name__ == "__main__":
    main()
