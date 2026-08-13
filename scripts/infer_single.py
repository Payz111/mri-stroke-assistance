"""Run inference on a single case.

Usage:
    python scripts/infer_single.py --input /path/to/subject --output /path/to/output \
        --checkpoint outputs/fold_0/checkpoints/best_model.pth

The input directory must contain one NIfTI per modality, with 'dwi', 'adc'
and 'flair' in their filenames (ISLES / BIDS naming).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import load_model, run_inference

MODALITIES = ("dwi", "adc", "flair")


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load a NIfTI file and return (data, affine, spacing)."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, img.affine, spacing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-case inference")
    parser.add_argument("--input", type=Path, required=True, help="Path to subject directory")
    parser.add_argument("--output", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML whose 'model' block must match the checkpoint architecture",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    return parser.parse_args()


def find_modality(subject_dir: Path, modality: str) -> Path:
    """Locate the NIfTI file for *modality* inside *subject_dir*."""
    matches = [
        p
        for p in sorted(subject_dir.rglob("*.nii*"))
        if modality in p.name.lower() and "mask" not in p.name.lower()
    ]
    if not matches:
        raise FileNotFoundError(f"No {modality.upper()} file found in {subject_dir}")
    return matches[0]


def main() -> None:
    args = parse_args()

    if not args.input.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    volumes = {}
    affine = None
    spacing = None
    for modality in MODALITIES:
        path = find_modality(args.input, modality)
        print(f"[load] {modality.upper():5s} {path}")
        data, aff, spc = load_nifti(path)
        volumes[modality] = data
        if modality == "dwi":
            affine, spacing = aff, spc

    metadata = {"subject_id": args.input.name, "spacing": spacing}

    model_config = None
    if args.model_config.is_file():
        model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))["model"]

    print(f"[model] loading checkpoint {args.checkpoint} on {args.device}")
    print(f"        architecture: {(model_config or {}).get('name', 'unet3d')}")
    model = load_model(args.checkpoint, config=model_config, device=args.device)

    print("[infer] running pipeline...")
    result = run_inference(
        model,
        dwi=volumes["dwi"],
        adc=volumes["adc"],
        flair=volumes["flair"],
        metadata=metadata,
        device=args.device,
        threshold=args.threshold,
    )

    report_path = args.output / "report.txt"
    report_path.write_text(result["report"], encoding="utf-8")

    # Quality control declined to predict (ADR-006): write the QC report and
    # exit non-zero, so a calling script can tell a refusal from a success.
    if result["pred_mask"] is None:
        qc_path = args.output / "qc_report.json"
        qc_path.write_text(json.dumps(result["qc"], indent=2), encoding="utf-8")
        print("\n[QC FAILED] no prediction produced")
        for reason in result["qc"]["critical_failures"]:
            print(f"       ! {reason}")
        print(f"\nWrote {qc_path.name} and {report_path.name} to {args.output}")
        sys.exit(2)

    mask_path = args.output / "prediction_mask.nii.gz"
    nib.save(nib.Nifti1Image(result["pred_mask"], affine), mask_path)

    findings_path = args.output / "findings.json"
    findings_path.write_text(json.dumps(result["findings"], indent=2), encoding="utf-8")

    if result["qc"] is not None:
        qc_path = args.output / "qc_report.json"
        qc_path.write_text(json.dumps(result["qc"], indent=2), encoding="utf-8")

    findings = result["findings"]
    validation = result["validation"]
    print(f"\n[done] QC passed; lesions: {findings['total_lesion_count']}")
    print(f"       volume: {findings['total_lesion_volume_ml']:.1f} mL")
    print(f"       report valid: {validation['valid']}")
    if not validation["valid"]:
        for issue in validation["issues"]:
            print(f"       ! {issue}")
    print(f"\nWrote {mask_path.name}, {findings_path.name}, {report_path.name} to {args.output}")


if __name__ == "__main__":
    main()
