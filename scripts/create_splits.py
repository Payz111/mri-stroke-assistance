"""Create 5-fold CV splits with stratification by lesion size.

Usage:
    python scripts/create_splits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def create_splits(
    metadata_path: Path,
    output_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """Create stratified K-fold splits and save to JSON files.

    Args:
        metadata_path: Path to metadata CSV
        output_dir: Directory to save split JSON files
        n_splits: Number of folds (default: 5)
        random_state: Random seed for reproducibility
    """
    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} cases from {metadata_path}")

    # Handle NaN in size_category (cases with 0 volume)
    # Assign them to "tiny" category for stratification
    if df["size_category"].isna().any():
        n_nan = df["size_category"].isna().sum()
        nan_cases = df[df["size_category"].isna()]["subject_id"].tolist()
        print(f"\nWarning: {n_nan} cases with NaN size_category (0 volume): {nan_cases}")
        print("  Assigning to 'tiny (<1ml)' category for stratification")
        df["size_category"] = df["size_category"].fillna("tiny (<1ml)")

    # Stratification variable
    stratify_var = df["size_category"].values
    subject_ids = df["subject_id"].values

    print("\nStratification distribution:")
    print(df["size_category"].value_counts().sort_index())

    # Create splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(subject_ids, stratify_var)):
        train_ids = subject_ids[train_indices].tolist()
        val_ids = subject_ids[val_indices].tolist()

        split_data = {
            "fold": fold_idx,
            "train": train_ids,
            "val": val_ids,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
        }

        output_file = output_dir / f"fold_{fold_idx}.json"
        with open(output_file, "w") as f:
            json.dump(split_data, f, indent=2)

        print(
            f"\nFold {fold_idx}: {len(train_ids)} train, {len(val_ids)} val -> {output_file.name}"
        )

        # Show stratification balance in validation set
        val_df = df[df["subject_id"].isin(val_ids)]
        val_dist = val_df["size_category"].value_counts().sort_index()
        print(f"  Val distribution: {val_dist.to_dict()}")

    print(f"\nDone! Created {n_splits} folds in {output_dir}")


def main() -> None:
    metadata_path = project_root / "data" / "processed" / "isles22_metadata.csv"
    output_dir = project_root / "data" / "splits"

    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Run EDA notebook first to generate metadata.")
        sys.exit(1)

    create_splits(metadata_path, output_dir)


if __name__ == "__main__":
    main()
