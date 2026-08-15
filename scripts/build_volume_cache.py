"""Build the packed volume cache from ISLES 2022 and SOOP.

The pack is fold-independent -- folds only change which subjects are training
and which are validation -- so this runs once and serves all five folds.

Usage:
    python scripts/build_volume_cache.py --output /kaggle/working/stroke_cache
    python scripts/build_volume_cache.py --output cache/ --isles-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.isles22_dataset import ISLES22Dataset
from src.data.soop_dataset import SOOPDataset
from src.data.volume_cache import build_pack

DEFAULT_ISLES = Path("data/raw/isles22/ISLES-2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the packed volume cache")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory")
    parser.add_argument("--isles-root", type=Path, default=DEFAULT_ISLES)
    parser.add_argument(
        "--isles-derivatives",
        type=Path,
        default=None,
        help="Defaults to <isles-root>/derivatives",
    )
    parser.add_argument("--soop-root", type=Path, default=None, help="ds004889 root")
    parser.add_argument("--isles-only", action="store_true", help="Skip SOOP")
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=(128, 128, 80),
        metavar=("D", "H", "W"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pack only the first N subjects per source -- a smoke run before committing 12 GB",
    )
    return parser.parse_args()


class _Subset:
    """First *n* subjects of a dataset, preserving the subject_ids contract."""

    def __init__(self, source, n: int) -> None:
        self.source = source
        self.subject_ids = list(source.subject_ids)[:n]

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int):
        return self.source[index]


def main() -> None:
    args = parse_args()

    derivatives = args.isles_derivatives or (args.isles_root / "derivatives")
    if not args.isles_root.is_dir():
        raise NotADirectoryError(f"ISLES root not found: {args.isles_root}")

    sources = []

    # Every ISLES subject, not a split: the cache serves all folds.
    isles = ISLES22Dataset(
        data_root=args.isles_root,
        derivatives_root=derivatives,
        split_file=None,
    )
    print(f"ISLES subjects: {len(isles)}")
    sources.append(isles)

    if not args.isles_only:
        if args.soop_root is None:
            raise SystemExit("Pass --soop-root, or --isles-only to build without SOOP.")
        soop = SOOPDataset(data_root=args.soop_root, require_mask=True)
        print(f"SOOP subjects : {len(soop)}")
        sources.append(soop)

    if args.limit is not None:
        sources = [_Subset(s, args.limit) for s in sources]
        print(f"Limited to the first {args.limit} subjects per source")

    total = sum(len(s) for s in sources)
    estimated_gb = total * 9.2 / 1000
    print(f"\nPacking {total} subjects at {tuple(args.spatial_size)} (~{estimated_gb:.1f} GB)")

    started = time.time()
    index = build_pack(sources, args.output, spatial_size=tuple(args.spatial_size))
    elapsed = time.time() - started

    print(f"\nDone in {elapsed / 60:.1f} min")
    print(f"  packed : {index['n_subjects']}")
    print(f"  skipped: {len(index['skipped'])}")
    for path in sorted(Path(args.output).iterdir()):
        print(f"  {path.name:14s} {path.stat().st_size / 1e9:6.2f} GB")


if __name__ == "__main__":
    main()
