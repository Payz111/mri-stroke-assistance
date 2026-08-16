"""Normalise a training run downloaded from Kaggle.

Kaggle output arrives in a shape that does not match the repository layout:

* ``best_model.pth`` is a zip archive, and downloading often expands it into a
  directory of the same name
* metadata and history carry a ``_fold{N}`` suffix
* the training curves are missing whenever the plotting cell did not run

This puts one run into the standard form::

    Train_<date>_fold<N>/
      checkpoints/best_model.pth
      experiment_meta.json
      training_history.json
      training_curves.png

Usage:
    python scripts/ingest_kaggle_run.py Training_results/Train_08_16_2026_fold1
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PREVIOUS_BEST = 0.785


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalise a Kaggle training run")
    parser.add_argument("run_dir", type=Path, help="Directory holding the downloaded output")
    parser.add_argument(
        "--keep-expanded",
        action="store_true",
        help="Keep the expanded best_model/ directory after rezipping",
    )
    return parser.parse_args()


def reassemble_checkpoint(run_dir: Path) -> Path:
    """Rebuild best_model.pth from an expanded directory, if that is what we got."""
    target = run_dir / "checkpoints" / "best_model.pth"
    if target.is_file():
        print(f"checkpoint already in place: {target.name}")
        return target

    flat = run_dir / "best_model.pth"
    expanded = run_dir / "best_model"

    target.parent.mkdir(parents=True, exist_ok=True)

    if flat.is_file():
        shutil.move(str(flat), target)
        print(f"moved {flat.name} -> checkpoints/")
        return target

    if not expanded.is_dir():
        raise FileNotFoundError(
            f"No checkpoint in {run_dir}: expected best_model.pth or an expanded best_model/"
        )

    files = sorted(p for p in expanded.rglob("*") if p.is_file())
    # A torch archive is a plain zip whose entries sit under one top-level
    # directory; ZIP_STORED because the tensors are already dense.
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, arcname=f"best_model/{path.relative_to(expanded).as_posix()}")
    print(
        f"reassembled {len(files)} files -> checkpoints/best_model.pth "
        f"({target.stat().st_size / 1e6:.1f} MB)"
    )
    return target


def normalise_names(run_dir: Path) -> tuple[Path, Path]:
    """Drop the _fold{N} suffixes Kaggle's flat copies carry."""
    result = []
    for stem in ("experiment_meta", "training_history"):
        target = run_dir / f"{stem}.json"
        if not target.is_file():
            matches = sorted(run_dir.glob(f"{stem}_fold*.json"))
            if not matches:
                raise FileNotFoundError(f"{stem}.json not found in {run_dir}")
            shutil.move(str(matches[0]), target)
            print(f"renamed {matches[0].name} -> {target.name}")
        result.append(target)
    return result[0], result[1]


def plot_curves(run_dir: Path, meta: dict, history: list[dict]) -> Path:
    epochs = [h["epoch"] + 1 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [h["train_dice"] for h in history], label="Train")
    axes[1].plot(epochs, [h["val_dice"] for h in history], label="Val")
    axes[1].axhline(
        PREVIOUS_BEST,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Published run ({PREVIOUS_BEST})",
    )
    axes[1].scatter(
        [meta["best_epoch"]],
        [meta["best_val_dice"]],
        color="red",
        zorder=5,
        label=f"Best: {meta['best_val_dice']:.4f} @ epoch {meta['best_epoch']}",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Dice score (batch-averaged)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Attention U-Net, fold {meta.get('fold', '?')} - ISLES 2022 + SOOP "
        f"({meta['train_subjects']} train / {meta['val_subjects']} val)"
    )
    fig.tight_layout()
    out = run_dir / "training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out.name} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    checkpoint = reassemble_checkpoint(run_dir)
    meta_path, history_path = normalise_names(run_dir)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    plot_curves(run_dir, meta, history)

    if not args.keep_expanded and (run_dir / "best_model").is_dir():
        shutil.rmtree(run_dir / "best_model")
        print("removed the expanded best_model/ directory")

    fold = meta.get("fold", "?")
    print(
        f"\nfold {fold}: best val_dice {meta['best_val_dice']:.4f} "
        f"at epoch {meta['best_epoch']} of {meta['epochs_trained']}"
    )
    print(f"train subjects: {meta['train_subjects']}, val: {meta['val_subjects']}")
    if meta["train_subjects"] < 1000:
        print("WARNING: fewer than 1000 training subjects -- SOOP may not have loaded.")

    print("\nNext:")
    print(f"  python scripts/evaluate.py --checkpoint {checkpoint} --fold {fold} \\")
    print(f"      --device cpu --output {run_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
