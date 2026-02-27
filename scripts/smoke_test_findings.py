"""Smoke test: findings extraction + report generation.

Loads one subject from the dataset, runs it through the findings
builder and report generator with synthetic / real predictions.
Can optionally load a trained checkpoint for real inference.

Usage:
    python scripts/smoke_test_findings.py [--checkpoint PATH] [--fold 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.findings.builder import build_findings
from src.report.generator import generate_report
from src.report.validator import validate_report


def _make_synthetic_data() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic test data (no real data needed)."""
    shape = (64, 64, 40)

    # Synthetic volumes
    dwi = np.random.randn(*shape).astype(np.float32) * 100 + 500
    adc = np.random.randn(*shape).astype(np.float32) * 200 + 800
    flair = np.random.randn(*shape).astype(np.float32) * 100 + 400

    # Create a synthetic lesion in the left hemisphere
    mask = np.zeros(shape, dtype=np.float32)
    mask[20:30, 25:35, 15:25] = 1.0  # ~1000 voxels

    # Make DWI bright and ADC dark in lesion (acute ischemia)
    dwi[mask > 0] = 1200.0
    adc[mask > 0] = 300.0
    # Keep FLAIR normal in lesion (mismatch positive)
    flair[mask > 0] = 450.0

    metadata = {
        "subject_id": "sub-synthetic-001",
        "spacing": (1.0, 1.0, 3.0),
    }

    return metadata, mask, dwi, adc, flair


def _load_real_subject(fold: int = 0):
    """Load a real subject from the dataset."""
    from src.data.isles22_dataset import ISLES22Dataset

    data_root = Path("data/raw/isles22/ISLES-2022")
    deriv_root = Path("data/raw/isles22/ISLES-2022/derivatives")
    split_file = Path(f"data/splits/fold_{fold}.json")

    if not data_root.exists() or not split_file.exists():
        return None

    dataset = ISLES22Dataset(
        data_root=data_root,
        derivatives_root=deriv_root,
        split_file=split_file,
        split="val",
        transform=None,
    )

    sample = dataset[0]
    return sample


def main():
    parser = argparse.ArgumentParser(description="Smoke test findings + report")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for real inference")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no dataset needed)")
    args = parser.parse_args()

    print("=" * 60)
    print("SMOKE TEST: Findings extraction + Report generation")
    print("=" * 60)

    if args.synthetic or args.checkpoint is None:
        print("\n[1/4] Using synthetic data...")
        metadata, mask, dwi, adc, flair = _make_synthetic_data()
        pred_mask = mask  # Use GT as "prediction" for synthetic test
        print(f"  Subject: {metadata['subject_id']}")
        print(f"  Shape: {dwi.shape}, Spacing: {metadata['spacing']}")
        print(f"  Lesion voxels: {int(mask.sum())}")
    else:
        print(f"\n[1/4] Loading real subject (fold {args.fold})...")
        sample = _load_real_subject(args.fold)
        if sample is None:
            print("  Dataset not found, falling back to synthetic data.")
            metadata, mask, dwi, adc, flair = _make_synthetic_data()
            pred_mask = mask
        else:
            metadata = sample["metadata"]
            dwi = sample["dwi"]
            adc = sample["adc"]
            flair = sample["flair"]
            pred_mask = sample["mask"]  # Use GT as pred for now
            print(f"  Subject: {metadata['subject_id']}")
            print(f"  Shape: {dwi.shape}")
            print(f"  GT lesion voxels: {int(pred_mask.sum())}")

    # Step 2: Build findings
    print("\n[2/4] Building structured findings...")
    findings = build_findings(
        lesion_masks=[pred_mask],
        dwi=dwi,
        adc=adc,
        flair=flair,
        metadata=metadata,
    )
    print(f"  Lesions found: {findings['total_lesion_count']}")
    print(f"  Total volume: {findings['total_lesion_volume_ml']:.2f} mL")
    print(f"  Impression: {findings['overall_impression']}")
    print(f"  Confidence: {findings['overall_confidence']:.0%}")

    for lesion in findings["lesions"]:
        print(f"  - Lesion {lesion['lesion_id']}: "
              f"{lesion['volume_ml']} mL, "
              f"{lesion['laterality']}, "
              f"{lesion['vascular_territory']}, "
              f"ADC low={lesion['adc_low_confirmed']}, "
              f"mismatch={lesion['dwi_flair_mismatch']}")

    # Step 3: Generate report
    print("\n[3/4] Generating report...")
    report_text = generate_report(findings)
    print("-" * 50)
    print(report_text)
    print("-" * 50)

    # Step 4: Validate report
    print("[4/4] Validating report against findings...")
    validation = validate_report(report_text, findings)
    print(f"  Valid: {validation['valid']}")
    for check_name, passed in validation["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"  WARNING: {issue}")

    # Save outputs
    output_dir = Path("outputs/smoke_test_findings")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    with open(output_dir / "validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\nOutputs saved to {output_dir}/")
    print("\nSMOKE TEST PASSED!" if validation["valid"] else "\nSMOKE TEST: validation issues found")
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
