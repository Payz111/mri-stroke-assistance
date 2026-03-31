"""Gradio demo app for MRI Stroke Assist.

Full pipeline: NIfTI upload -> preprocessing -> model inference ->
structured findings (JSON) -> radiology report + visualization.

Supports two upload modes:
1. Single ZIP/folder with all modalities (auto-detects DWI/ADC/FLAIR)
2. Three separate NIfTI files
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
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


def _find_nifti_by_keyword(base: Path, keyword: str) -> Path | None:
    """Find a NIfTI file containing keyword in its name.

    Reuses the same logic as src/data/isles22_dataset._find_nifti.
    """
    kw = keyword.lower()
    # Try .nii.gz first
    for p in base.rglob("*.nii.gz"):
        if kw in p.name.lower() and p.is_file() and p.stat().st_size > 0:
            return p
    # Fall back to .nii
    for p in base.rglob("*.nii"):
        if kw in p.name.lower() and p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _extract_archive(archive_path: str) -> Path:
    """Extract ZIP archive to a temp directory, return the path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="mri_stroke_"))
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(tmp_dir)
    return tmp_dir


def _find_modalities_in_dir(base_dir: Path) -> dict[str, Path]:
    """Auto-detect DWI, ADC, FLAIR files in a directory."""
    dwi = _find_nifti_by_keyword(base_dir, "dwi")
    adc = _find_nifti_by_keyword(base_dir, "adc")
    flair = _find_nifti_by_keyword(base_dir, "flair")

    found = {}
    missing = []
    if dwi:
        found["dwi"] = dwi
    else:
        missing.append("DWI")
    if adc:
        found["adc"] = adc
    else:
        missing.append("ADC")
    if flair:
        found["flair"] = flair
    else:
        missing.append("FLAIR")

    if missing:
        all_nifti = list(base_dir.rglob("*.nii*"))
        names = [p.name for p in all_nifti[:10]]
        raise FileNotFoundError(
            f"Could not find: {', '.join(missing)}.\n"
            f"Files in archive: {names}\n"
            "Files must contain 'dwi', 'adc', or 'flair' in their names."
        )
    return found


def _run_analysis(dwi_path, adc_path, flair_path):
    """Core analysis: load files, run inference, return results."""
    from src.inference.pipeline import run_inference
    from src.inference.visualize import create_montage

    dwi, dwi_img = _load_nifti(str(dwi_path))
    adc, _ = _load_nifti(str(adc_path))
    flair, _ = _load_nifti(str(flair_path))

    subject_name = dwi_path.parent.name or "uploaded"
    metadata = {
        "subject_id": subject_name,
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
    findings_str = json.dumps(findings, default=str, indent=2)

    return report, findings_str, preview


def predict_archive(archive_file):
    """Run analysis on a ZIP archive containing DWI/ADC/FLAIR."""
    if archive_file is None:
        return "Upload a ZIP archive with DWI, ADC, and FLAIR files.", None, None

    tmp_dir = None
    try:
        tmp_dir = _extract_archive(archive_file)
        paths = _find_modalities_in_dir(tmp_dir)

        status = (
            f"Found: DWI={paths['dwi'].name}, "
            f"ADC={paths['adc'].name}, "
            f"FLAIR={paths['flair'].name}\n"
        )
        report, findings, preview = _run_analysis(
            paths["dwi"], paths["adc"], paths["flair"]
        )
        return status + report, findings, preview

    except FileNotFoundError as e:
        return f"Error: {e}", None, None
    except zipfile.BadZipFile:
        return "Error: File is not a valid ZIP archive.", None, None
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", None, None
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def predict_separate(dwi_file, adc_file, flair_file):
    """Run analysis on 3 separately uploaded NIfTI files."""
    if dwi_file is None or adc_file is None or flair_file is None:
        return "Upload all 3 files: DWI, ADC, FLAIR.", None, None

    try:
        return _run_analysis(Path(dwi_file), Path(adc_file), Path(flair_file))
    except FileNotFoundError as e:
        return f"Error: {e}", None, None
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", None, None


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
        findings_str = json.dumps(findings, default=str, indent=2)

        return report, findings_str, preview

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

            > **Disclaimer:** Research tool only. All results require expert review.
            """
        )

        with gr.Tabs():
            # Tab 1: ZIP archive (simplest for user)
            with gr.Tab("Upload ZIP archive"):
                gr.Markdown(
                    "Upload a **single ZIP file** containing DWI, ADC, and "
                    "FLAIR NIfTI files. The app will find them automatically "
                    "by filename (must contain 'dwi', 'adc', 'flair')."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        zip_input = gr.File(
                            label="ZIP archive with MRI data",
                            file_types=[".zip"],
                        )
                        with gr.Row():
                            zip_btn = gr.Button(
                                "Analyze", variant="primary", size="lg"
                            )
                            demo_btn = gr.Button(
                                "Synthetic Demo", variant="secondary", size="lg"
                            )

                    with gr.Column(scale=2):
                        zip_preview = gr.Image(
                            label="Lesion Overlay", height=400
                        )
                        zip_report = gr.Textbox(
                            label="Draft Radiology Report", lines=18
                        )

                zip_json = gr.Code(label="Structured Findings (JSON)", language="json")

                zip_btn.click(
                    fn=predict_archive,
                    inputs=[zip_input],
                    outputs=[zip_report, zip_json, zip_preview],
                )
                demo_btn.click(
                    fn=predict_synthetic,
                    inputs=[],
                    outputs=[zip_report, zip_json, zip_preview],
                )

            # Tab 2: Separate files
            with gr.Tab("Upload files separately"):
                gr.Markdown(
                    "Upload each modality as a separate NIfTI file."
                )
                with gr.Row():
                    with gr.Column(scale=1):
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
                        sep_btn = gr.Button(
                            "Analyze", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        sep_preview = gr.Image(
                            label="Lesion Overlay", height=400
                        )
                        sep_report = gr.Textbox(
                            label="Draft Radiology Report", lines=18
                        )

                sep_json = gr.Code(label="Structured Findings (JSON)", language="json")

                sep_btn.click(
                    fn=predict_separate,
                    inputs=[dwi_input, adc_input, flair_input],
                    outputs=[sep_report, sep_json, sep_preview],
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
