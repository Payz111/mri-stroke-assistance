"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --config-name default
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    raise NotImplementedError("Evaluation pipeline not yet implemented")


if __name__ == "__main__":
    main()
