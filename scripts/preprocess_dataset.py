"""Preprocess entire dataset.

Usage:
    python scripts/preprocess_dataset.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise NotImplementedError("Preprocessing pipeline not yet implemented")


if __name__ == "__main__":
    main()
