"""Gradio demo app for MRI Stroke Assist.

Full pipeline: NIfTI upload -> preprocessing -> model inference ->
structured findings (JSON) -> radiology report + visualization.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np

# Lazy-loaded globals
_model = None
_device = "cpu"

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "fold_0" / "checkpoints" / "best_model.pth"


def _get_model():
    """Load model lazily on first request."""
    global _model
    if _model is not None:
        return _model

    from src.inference.pipeline import load_model

    if not DEFAULT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {DEFAULT_CHECKPOINT}\n"
            "Please place best_model.pth in outputs/fold_0/checkpoints/"
        )

    _model = load_model(DEFAULT_CHECKPOINT, device=_device)
    return _model


def _load_nifti(file_path: str) -> tuple:
    """Load a NIfTI file and return (data, nib_image)."""
    import nibabel as nib

    img = nib.load(file_path)
    data = img.get_fdata(dtype=np.float32)
    return data, img


def predict(dwi_file, adc_file, flair_file):
    """Run full inference pipeline on uploaded NIfTI files."""
    if dwi_file is None or adc_file is None or flair_file is None:
        return "Error: Please upload all 3 files (DWI, ADC, FLAIR).", None, None

    try:
        from src.inference.pipeline import run_inference
        from src.inference.visualize import create_montage

        dwi, dwi_img = _load_nifti(dwi_file)
        adc, _ = _load_nifti(adc_file)
        flair, _ = _load_nifti(flair_file)

        metadata = {
            "subject_id": Path(dwi_file).parent.name or "uploaded",
            "spacing": tuple(float(s) for s in dwi_img.header.get_zooms()[:3]),
            "affine": dwi_img.affine,
            "shape": dwi.shape,
        }

        model = _get_model()
        result = run_inference(
            model=model,
            dwi=dwi,
            adc=adc,
            flair=flair,
            metadata=metadata,
            device=_device,
        )

        report = result["report"]
        findings = result["findings"]
        validation = result["validation"]

        if not validation["valid"]:
            issues = "; ".join(validation["issues"])
            report += f"\n\n*** VALIDATION WARNING: {issues} ***"

        preview = create_montage(dwi, result["pred_mask"], n_slices=6)
        findings_clean = json.loads(json.dumps(findings, default=str))

        return report, findings_clean, preview

    except FileNotFoundError as e:
        return f"Error: {e}", None, None
    except Exception as e:
        return f"Error during inference: {type(e).__name__}: {e}", None, None


def predict_synthetic():
    """Run inference on synthetic data (no files or checkpoint needed)."""
    try:
        from src.findings.builder import build_findings
        from src.inference.visualize import create_overlay_image
        from src.report.generator import generate_report

        shape = (64, 64, 40)
        dwi = np.random.randn(*shape).astype(np.float32) * 100 + 500
        adc = np.random.randn(*shape).astype(np.float32) * 200 + 800
        flair = np.random.randn(*shape).astype(np.float32) * 100 + 400

        mask = np.zeros(shape, dtype=np.float32)
        mask[20:30, 25:35, 15:25] = 1.0
        dwi[mask > 0] = 1200.0
        adc[mask > 0] = 300.0

        metadata = {
            "subject_id": "synthetic-demo",
            "spacing": (1.0, 1.0, 3.0),
        }

        findings = build_findings([mask], dwi, adc, flair, metadata)
        report = generate_report(findings)
        preview = create_overlay_image(dwi, mask)
        findings_clean = json.loads(json.dumps(findings, default=str))

        return report, findings_clean, preview

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", None, None


def create_app():
    """Create and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Install gradio: pip install gradio")

    with gr.Blocks(title="MRI Stroke Assist") as app:
        gr.Markdown(
            """
            # MRI Stroke Assist
            AI-powered assistant for ischemic stroke detection on brain MRI.

            **Upload DWI, ADC, and FLAIR NIfTI files** to get:
            - Automated lesion segmentation
            - Structured findings (JSON)
            - Draft radiology report

            > **Disclaimer:** This is a research tool for educational purposes only.
            > All results require expert review. Not for clinical use.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                dwi_input = gr.File(
                    label="DWI (.nii / .nii.gz)",
                    file_types=[".nii", ".nii.gz", ".gz"],
                )
                adc_input = gr.File(
                    label="ADC (.nii / .nii.gz)",
                    file_types=[".nii", ".nii.gz", ".gz"],
                )
                flair_input = gr.File(
                    label="FLAIR (.nii / .nii.gz)",
                    file_types=[".nii", ".nii.gz", ".gz"],
                )

                with gr.Row():
                    run_btn = gr.Button(
                        "Run Analysis", variant="primary", size="lg"
                    )
                    demo_btn = gr.Button(
                        "Synthetic Demo", variant="secondary", size="lg"
                    )

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                preview_output = gr.Image(label="Lesion Overlay", height=400)
                report_output = gr.Textbox(
                    label="Draft Radiology Report",
                    lines=18,
                )

        with gr.Row():
            json_output = gr.JSON(label="Structured Findings (JSON)")

        run_btn.click(
            fn=predict,
            inputs=[dwi_input, adc_input, flair_input],
            outputs=[report_output, json_output, preview_output],
        )

        demo_btn.click(
            fn=predict_synthetic,
            inputs=[],
            outputs=[report_output, json_output, preview_output],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
