"""End-to-end inference pipeline.

Takes raw NIfTI volumes, preprocesses them, runs the segmentation model,
applies post-processing, builds findings, and generates a report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.transforms import get_val_transforms
from src.findings.builder import build_findings
from src.models.factory import create_model
from src.report.generator import generate_report
from src.report.validator import validate_report


def load_model(
    checkpoint_path: str | Path,
    config: dict[str, Any] | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load the UNet3D model with trained weights.

    Parameters
    ----------
    checkpoint_path:
        Path to the saved model checkpoint (.pth file).
    config:
        Model configuration dict. If None, uses default config.
    device:
        Device to load model onto ("cpu" or "cuda").

    Returns
    -------
    torch.nn.Module
        Model in eval mode, ready for inference.
    """
    if config is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        config = full_config["model"]

    model = create_model(config)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_mask(
    model: torch.nn.Module,
    dwi: np.ndarray,
    adc: np.ndarray,
    flair: np.ndarray,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """Run model inference and return the predicted binary mask.

    Parameters
    ----------
    model:
        Loaded model in eval mode.
    dwi, adc, flair:
        Raw 3-D volumes (D, H, W) as float32 numpy arrays.
    device:
        Device for inference.
    threshold:
        Probability threshold for binarization.

    Returns
    -------
    np.ndarray
        Binary prediction mask with the same shape as the input DWI.
    """
    original_shape = dwi.shape

    # Build sample dict with dummy mask for transforms
    sample = {
        "dwi": dwi,
        "adc": adc,
        "flair": flair,
        "mask": np.zeros_like(dwi),
        "metadata": {},
    }

    transforms = get_val_transforms()
    processed = transforms(sample)

    image = processed["image"].unsqueeze(0).to(device)  # (1, 3, D, H, W)

    with torch.no_grad():
        logits = model(image)
        pred_prob = torch.sigmoid(logits)
        pred_binary = (pred_prob > threshold).float()

    pred_np = pred_binary[0, 0].cpu().numpy()  # (D, H, W)

    # Crop back to original spatial size (transforms may have padded)
    pred_np = pred_np[
        : original_shape[0],
        : original_shape[1],
        : original_shape[2],
    ]

    return pred_np.astype(np.float32)


def run_inference(
    model: torch.nn.Module,
    dwi: np.ndarray,
    adc: np.ndarray,
    flair: np.ndarray,
    metadata: dict[str, Any],
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the full inference pipeline on a single subject.

    Parameters
    ----------
    model:
        Loaded model in eval mode.
    dwi, adc, flair:
        Raw 3-D volumes (D, H, W) as float32.
    metadata:
        Subject metadata (must include ``spacing``, ``subject_id``).
    device:
        Device for inference.
    threshold:
        Probability threshold.

    Returns
    -------
    dict
        Keys: ``pred_mask``, ``findings``, ``report``, ``validation``.
    """
    # Step 1: Model prediction
    pred_mask = predict_mask(model, dwi, adc, flair, device, threshold)

    # Step 2: Build structured findings
    findings = build_findings(
        lesion_masks=[pred_mask],
        dwi=dwi,
        adc=adc,
        flair=flair,
        metadata=metadata,
    )

    # Step 3: Generate report
    report = generate_report(findings)

    # Step 4: Validate
    validation = validate_report(report, findings)

    return {
        "pred_mask": pred_mask,
        "findings": findings,
        "report": report,
        "validation": validation,
    }
