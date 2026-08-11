"""Preprocess an entire dataset into .npz files.

Usage:
    python scripts/preprocess_dataset.py --output data/processed/npz
    python scripts/preprocess_dataset.py --output data/processed/npz --split data/splits/fold_0.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess.pipeline import preprocess_dataset

DEFAULT_DATA_ROOT = Path("data/raw/isles22/ISLES-2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--derivatives-root",
        type=Path,
        default=None,
        help="Defaults to <data-root>/derivatives",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination for .npz files")
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="Optional split JSON; only its train/val subjects are processed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    derivatives_root = args.derivatives_root or (args.data_root / "derivatives")
    if not args.data_root.is_dir():
        raise NotADirectoryError(f"Data root not found: {args.data_root}")

    config = None
    if args.config.is_file():
        full_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        config = (full_config or {}).get("preprocessing")

    print(f"[data]   {args.data_root}")
    print(f"[deriv]  {derivatives_root}")
    print(f"[output] {args.output}")
    if args.split:
        print(f"[split]  {args.split}")

    preprocess_dataset(
        data_root=args.data_root,
        derivatives_root=derivatives_root,
        output_dir=args.output,
        split_file=args.split,
        config=config,
    )

    n_files = len(list(args.output.glob("*.npz")))
    print(f"\n[done] {n_files} .npz files in {args.output}")


if __name__ == "__main__":
    main()
