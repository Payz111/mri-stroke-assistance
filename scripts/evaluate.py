"""Evaluation entry point.

Usage:
    python scripts/evaluate.py --checkpoint outputs/fold_0/checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint best_model.pth --fold 0 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.isles22_dataset import ISLES22Dataset
from src.data.transforms import get_val_transforms
from src.eval.metrics import compute_all_metrics
from src.eval.stratified import stratified_evaluation
from src.models.factory import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on val set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--fold", type=int, default=0, help="CV fold index")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Load config
    cfg_path = Path(args.config) if args.config else project_root / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Paths
    data_root = project_root / cfg["paths"]["data_raw"] / "ISLES-2022"
    derivatives_root = data_root / "derivatives"
    split_file = project_root / cfg["paths"]["splits"] / f"fold_{args.fold}.json"

    # Load model
    model = create_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    logging.info("Model loaded from %s", args.checkpoint)

    # Load val dataset (without transforms for raw masks)
    val_ds_raw = ISLES22Dataset(
        data_root=data_root,
        derivatives_root=derivatives_root,
        split_file=split_file,
        split="val",
    )

    # Load val dataset (with transforms for model input)
    val_ds = ISLES22Dataset(
        data_root=data_root,
        derivatives_root=derivatives_root,
        split_file=split_file,
        split="val",
        transform=get_val_transforms(),
    )

    logging.info("Evaluating %d subjects...", len(val_ds))

    predictions = []
    ground_truths = []
    metadata_list = []

    for i in range(len(val_ds)):
        sample = val_ds[i]
        raw_sample = val_ds_raw[i]

        image = sample["image"].unsqueeze(0).to(device)
        # Use transformed GT at model resolution (128x128x80)
        gt_tensor = sample["label"]

        with torch.no_grad():
            logits = model(image)
            pred_prob = torch.sigmoid(logits)
            pred_mask = (pred_prob > 0.5).float()

        pred_np = pred_mask[0, 0].cpu().numpy()
        gt_np = (gt_tensor[0].numpy() > 0).astype(np.float32)

        predictions.append(pred_np)
        ground_truths.append(gt_np)
        metadata_list.append(raw_sample["metadata"])

        sid = raw_sample["metadata"]["subject_id"]
        metrics = compute_all_metrics(
            pred_np, gt_np, tuple(float(s) for s in raw_sample["metadata"]["spacing"])
        )
        if (i + 1) % 10 == 0 or i == 0:
            logging.info(
                "[%d/%d] %s: Dice=%.4f, HD95=%.2f, Vol MAE=%.2f ml",
                i + 1,
                len(val_ds),
                sid,
                metrics["dice"],
                metrics["hd95"],
                metrics["volume_mae_ml"],
            )

    # Stratified evaluation
    results = stratified_evaluation(predictions, ground_truths, metadata_list)

    # Print summary
    overall = results["overall"]
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Subjects: {overall['n_subjects']}")
    print(f"Dice:     {overall.get('dice_mean', 0):.4f} +/- {overall.get('dice_std', 0):.4f}")
    print(f"IoU:      {overall.get('iou_mean', 0):.4f} +/- {overall.get('iou_std', 0):.4f}")
    print(f"HD95:     {overall.get('hd95_mean', 0):.2f} +/- {overall.get('hd95_std', 0):.2f} mm")
    vol_mae_mean = overall.get("volume_mae_ml_mean", 0)
    vol_mae_std = overall.get("volume_mae_ml_std", 0)
    print(f"Vol MAE:  {vol_mae_mean:.2f} +/- {vol_mae_std:.2f} ml")
    print(f"Lesion F1:{overall.get('lesion_f1_mean', 0):.4f}")
    print(f"Sensitivity: {overall.get('sensitivity_mean', 0):.4f}")

    print("\nBy lesion size:")
    for cat, cat_data in results["by_size"].items():
        n = cat_data.get("n", 0)
        dice = cat_data.get("dice_mean", 0)
        print(f"  {cat:20s}: n={n:3d}, Dice={dice:.4f}")

    # Save results
    output_path = args.output or str(
        project_root / cfg["paths"]["outputs"] / f"fold_{args.fold}" / "eval_results.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable types
    serializable_results = {
        "overall": results["overall"],
        "by_size": results["by_size"],
        "per_subject": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in s.items()}
            for s in results["per_subject"]
        ],
    }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logging.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
