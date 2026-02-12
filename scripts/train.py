"""Training entry point.

Usage:
    python scripts/train.py --config-name baseline
    python scripts/train.py --config-name baseline training.fold=0
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    # Hydra will be configured here
    raise NotImplementedError("Training pipeline not yet implemented")


if __name__ == "__main__":
    main()
