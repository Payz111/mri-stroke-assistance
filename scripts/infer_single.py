"""Run inference on a single case.

Usage:
    python scripts/infer_single.py --input /path/to/subject --output /path/to/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-case inference")
    parser.add_argument("--input", type=Path, required=True, help="Path to subject directory")
    parser.add_argument("--output", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--config", type=str, default="default", help="Config name")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise NotImplementedError("Inference pipeline not yet implemented")


if __name__ == "__main__":
    main()
