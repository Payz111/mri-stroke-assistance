"""FastAPI application for stroke-lesion segmentation inference.

Exposes endpoints:
- POST /predict  — upload ZIP with DWI/ADC/FLAIR NIfTI, get findings + report
- GET  /health   — health check (model status, device)
- GET  /         — API info
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(
        Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "fold_0"
        / "checkpoints"
        / "best_model.pth"
    ),
)
DEVICE = os.environ.get("DEVICE", "cpu")
MODEL_NAME = os.environ.get("MODEL_NAME", "attention_unet3d")

# ---------------------------------------------------------------------------
# Global model (lazy-loaded)
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Load model on first request."""
    global _model
    if _model is not None:
        return _model

    from src.inference.pipeline import load_model

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        raise RuntimeError(
            f"Checkpoint not found: {ckpt}. "
            "Set CHECKPOINT_PATH env var or mount the checkpoint volume."
        )

    config = {
        "name": MODEL_NAME,
        "in_channels": 3,
        "out_channels": 1,
        "features": [32, 64, 128, 256],
        "dropout": 0.1,
    }
    _model = load_model(ckpt, config=config, device=DEVICE)
    logger.info("Model loaded: %s on %s", MODEL_NAME, DEVICE)
    return _model


# ---------------------------------------------------------------------------
# Pydantic response models (for OpenAPI docs)
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    checkpoint: str


class InfoResponse(BaseModel):
    name: str
    version: str
    description: str
    model: str
    endpoints: dict[str, str]


class PredictResponse(BaseModel):
    status: str
    findings: dict[str, Any]
    report: str
    validation: dict[str, Any]
    processing_time_s: float


class ErrorResponse(BaseModel):
    status: str
    detail: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MRI Stroke Assistance API",
    description=(
        "Automated ischaemic-stroke lesion segmentation and reporting. "
        "Upload a ZIP archive with DWI, ADC, and FLAIR NIfTI files."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from demo/app.py)
# ---------------------------------------------------------------------------
def _find_nifti_by_keyword(base: Path, keyword: str) -> Path | None:
    """Find a NIfTI file containing *keyword* in its name."""
    kw = keyword.lower()
    for ext in ("*.nii.gz", "*.nii"):
        for p in base.rglob(ext):
            if kw in p.name.lower() and p.is_file() and p.stat().st_size > 0:
                return p
    return None


def _find_modalities(base_dir: Path) -> dict[str, Path]:
    """Auto-detect DWI, ADC, FLAIR NIfTI files in a directory."""
    modalities = {}
    missing = []
    for key in ("dwi", "adc", "flair"):
        path = _find_nifti_by_keyword(base_dir, key)
        if path:
            modalities[key] = path
        else:
            missing.append(key.upper())

    if missing:
        all_nifti = [p.name for p in base_dir.rglob("*.nii*")][:10]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing modalities: {', '.join(missing)}. "
                f"Files in archive: {all_nifti}. "
                "Filenames must contain 'dwi', 'adc', or 'flair'."
            ),
        )
    return modalities


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_model=InfoResponse)
async def root():
    """API info and available endpoints."""
    return InfoResponse(
        name="MRI Stroke Assistance API",
        version="1.0.0",
        description="Ischaemic stroke lesion segmentation and structured reporting",
        model=MODEL_NAME,
        endpoints={
            "POST /predict": "Upload ZIP with DWI/ADC/FLAIR, get findings + report",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — reports model loading status."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=DEVICE,
        checkpoint=CHECKPOINT_PATH,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict(
    file: UploadFile = File(..., description="ZIP archive containing DWI, ADC, FLAIR NIfTI files"),
):
    """Run stroke lesion segmentation on uploaded MRI data.

    Upload a ZIP archive containing three NIfTI files (.nii or .nii.gz)
    with 'dwi', 'adc', and 'flair' in their filenames.

    Returns structured findings (JSON), a draft radiology report,
    and validation results.
    """
    import json

    import nibabel as nib

    from src.inference.pipeline import run_inference

    tmp_dir = None
    t0 = time.time()

    try:
        # 1. Save uploaded file to temp location
        tmp_dir = Path(tempfile.mkdtemp(prefix="stroke_api_"))
        zip_path = tmp_dir / "upload.zip"
        with open(zip_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 2. Extract ZIP
        if not zipfile.is_zipfile(zip_path):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir / "extracted")

        # 3. Find modalities
        modalities = _find_modalities(tmp_dir / "extracted")

        # 4. Load NIfTI data
        dwi_img = nib.load(str(modalities["dwi"]))
        dwi = dwi_img.get_fdata(dtype=np.float32)
        adc = nib.load(str(modalities["adc"])).get_fdata(dtype=np.float32)
        flair = nib.load(str(modalities["flair"])).get_fdata(dtype=np.float32)

        metadata = {
            "subject_id": file.filename or "api-upload",
            "spacing": tuple(float(s) for s in dwi_img.header.get_zooms()[:3]),
            "shape": dwi.shape,
        }

        # 5. Run inference
        model = _get_model()
        result = run_inference(
            model=model,
            dwi=dwi,
            adc=adc,
            flair=flair,
            metadata=metadata,
            device=DEVICE,
        )

        elapsed = time.time() - t0

        # 6. Serialize findings (handle numpy/datetime types)
        findings_clean = json.loads(json.dumps(result["findings"], default=str))

        return PredictResponse(
            status="success",
            findings=findings_clean,
            report=result["report"],
            validation=result["validation"],
            processing_time_s=round(elapsed, 2),
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
