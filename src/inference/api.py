"""FastAPI application for stroke-lesion segmentation inference.

Exposes a ``/predict`` POST endpoint that accepts NIfTI uploads and
returns structured findings plus a textual radiology report.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(
    title="MRI Stroke Assistance API",
    description="Automated ischaemic-stroke lesion segmentation and reporting.",
    version="0.1.0",
)


@app.post("/predict", response_class=JSONResponse)
async def predict(
    file: UploadFile = File(..., description="NIfTI (.nii/.nii.gz) archive to analyse"),
) -> dict[str, Any]:
    """Run inference on an uploaded NIfTI volume.

    Parameters
    ----------
    file:
        Uploaded NIfTI file containing the subject's MRI data.

    Returns
    -------
    dict
        JSON response with keys:
            - ``findings``: structured V1Findings dictionary
            - ``report``: generated radiology report text
            - ``status``: ``"success"`` or ``"error"``
    """
    raise NotImplementedError()
