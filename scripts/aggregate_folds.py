"""Aggregate per-fold evaluation results into mean +/- std.

A single fold's number invites the obvious objection -- was it luck? This
reports the spread across folds, which is the honest answer, and flags metrics
whose spread is large enough that the single-fold figure should not be quoted
on its own.

Usage:
    python scripts/aggregate_folds.py Training_results/Train_*_fold*/eval_results.json
    python scripts/aggregate_folds.py --glob "Training_results/*fold*/eval_results.json"
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import statistics
from pathlib import Path

OVERALL = [
    ("Dice (mean)", "dice_mean"),
    ("Dice (median)", "dice_median"),
    ("IoU", "iou_mean"),
    ("HD95 (mm)", "hd95_mean"),
    ("Volume MAE (mL)", "volume_mae_ml_mean"),
    ("Lesion F1", "lesion_f1_mean"),
    ("Sensitivity", "sensitivity_mean"),
]
# Keys carry their bounds, e.g. "tiny (<1ml)"
SIZES = ["tiny", "small", "medium", "large"]

# Above this coefficient of variation, a single-fold figure is not
# representative and should always be quoted with its spread.
UNSTABLE_CV = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-fold evaluation results")
    parser.add_argument("results", nargs="*", type=Path, help="eval_results.json files")
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern instead of paths")
    parser.add_argument("--markdown", action="store_true", help="Emit a markdown table")
    return parser.parse_args()


def summarise(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main() -> None:
    args = parse_args()

    paths = list(args.results)
    if args.glob:
        paths += [Path(p) for p in sorted(globlib.glob(args.glob))]
    if not paths:
        raise SystemExit("No eval_results.json given. Pass paths or --glob.")

    runs = []
    for path in sorted(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta_path = path.parent / "experiment_meta.json"
        fold = "?"
        if meta_path.is_file():
            fold = json.loads(meta_path.read_text(encoding="utf-8")).get("fold", "?")
        runs.append({"fold": fold, "dir": path.parent.name, "data": data})

    print(f"Folds aggregated: {len(runs)}")
    for run in runs:
        print(f"  fold {run['fold']}: {run['dir']}")
    print()

    header = f"{'metric':<18}" + "".join(f"{'fold ' + str(r['fold']):>10}" for r in runs)
    header += f"{'mean':>10}{'std':>9}"
    print(header)
    print("-" * len(header))

    unstable = []
    rows_md = []
    for label, key in OVERALL:
        values = [r["data"]["overall"].get(key) for r in runs]
        if any(v is None for v in values):
            continue
        mean, std = summarise(values)
        line = f"{label:<18}" + "".join(f"{v:>10.4f}" for v in values)
        print(line + f"{mean:>10.4f}{std:>9.4f}")
        rows_md.append((label, mean, std))
        if mean and abs(std / mean) > UNSTABLE_CV:
            unstable.append((label, mean, std))

    print()
    print("By lesion size (Dice):")
    size_header = f"{'category':<18}" + "".join(f"{'fold ' + str(r['fold']):>10}" for r in runs)
    print(size_header + f"{'mean':>10}{'std':>9}")
    print("-" * len(size_header + f"{'mean':>10}{'std':>9}"))
    for size in SIZES:
        values, counts = [], []
        for run in runs:
            by_size = run["data"].get("by_size", {})
            entry = next(
                (v for k, v in by_size.items() if k.split()[0] == size), None
            )
            if not entry:
                continue
            values.append(entry.get("dice_mean", entry.get("dice")))
            counts.append(entry.get("n", 0))
        if len(values) != len(runs) or any(v is None for v in values):
            continue
        mean, std = summarise(values)
        line = f"{size:<18}" + "".join(f"{v:>10.4f}" for v in values)
        print(line + f"{mean:>10.4f}{std:>9.4f}   n={counts}")
        if mean and abs(std / mean) > UNSTABLE_CV:
            unstable.append((f"Dice, {size} lesions", mean, std))

    if unstable:
        print()
        print(f"Unstable across folds (std/mean > {UNSTABLE_CV:.0%}) -- never quote alone:")
        for label, mean, std in unstable:
            print(f"  {label:<26} {mean:.4f} +/- {std:.4f}")

    if args.markdown:
        print()
        print("| Metric | Value |")
        print("|--------|-------|")
        for label, mean, std in rows_md:
            print(f"| {label} | {mean:.3f} ± {std:.3f} |")


if __name__ == "__main__":
    main()
