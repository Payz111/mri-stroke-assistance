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
from scipy.ndimage import zoom

from src.data.transforms import get_val_transforms
from src.findings.builder import build_findings
from src.models.factory import create_model
from src.qc.qc_pipeline import QCPipeline
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
    run_qc: bool = True,
    qc_config: dict[str, Any] | None = None,
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
    run_qc:
        Apply the quality-control gate first. On a critical failure the
        pipeline declines to predict (ADR-006) and returns ``pred_mask``,
        ``findings`` and ``validation`` as None, with ``report`` explaining
        why. Set False only to inspect raw model behaviour.
    qc_config:
        Threshold overrides for :class:`~src.qc.qc_pipeline.QCPipeline`.

    Returns
    -------
    dict
        Keys: ``qc``, ``pred_mask``, ``findings``, ``report``, ``validation``.
        Every key is always present; the last four are None when QC blocks.
    """
    # Step 0: Quality control -- refuse rather than guess on unreadable input
    qc_report = None
    if run_qc:
        qc_report = QCPipeline(qc_config).evaluate(
            {
                "dwi": dwi,
                "adc": adc,
                "flair": flair,
                "spacing": metadata.get("spacing"),
                "metadata": metadata,
            }
        )
        if not qc_report.passed:
            return {
                "qc": qc_report.to_dict(),
                "pred_mask": None,
                "findings": None,
                "report": qc_report.summary_text(),
                "validation": None,
            }

    # Step 1: Model prediction
    pred_mask = predict_mask(model, dwi, adc, flair, device, threshold)

    # Step 1.5: Resample modalities to DWI space for findings extraction
    # (FLAIR often has different resolution, e.g. 352x352x230 vs DWI 112x112x72)
    ref_shape = dwi.shape
    adc_resampled = adc
    flair_resampled = flair
    if adc.shape != ref_shape:
        factors = [r / s for r, s in zip(ref_shape, adc.shape)]
        adc_resampled = zoom(adc, factors, order=1).astype(np.float32)
    if flair.shape != ref_shape:
        factors = [r / s for r, s in zip(ref_shape, flair.shape)]
        flair_resampled = zoom(flair, factors, order=1).astype(np.float32)

    # Step 2: Build structured findings
    findings = build_findings(
        lesion_masks=[pred_mask],
        dwi=dwi,
        adc=adc_resampled,
        flair=flair_resampled,
        metadata=metadata,
    )

    # Step 3: Generate report
    report = generate_report(findings)

    # Step 4: Validate
    validation = validate_report(report, findings)

    return {
        "qc": qc_report.to_dict() if qc_report is not None else None,
        "pred_mask": pred_mask,
        "findings": findings,
        "report": report,
        "validation": validation,
    }
