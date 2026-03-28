# =============================================================================
# Multi-stage Dockerfile for MRI Stroke Assistance
# Targets: api (FastAPI) | demo (Gradio)
#
# Build:
#   docker build --target api  -t stroke-api  .
#   docker build --target demo -t stroke-demo .
#
# Run:
#   docker run -p 8000:8000 -v ./outputs:/app/outputs stroke-api
#   docker run -p 7860:7860 -v ./outputs:/app/outputs stroke-demo
# =============================================================================

# ---------------------------------------------------------------------------
# Base stage: shared dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for scipy, matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (smaller image, ~800 MB vs ~2 GB with CUDA)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Core Python dependencies
RUN pip install --no-cache-dir \
    monai>=1.3 \
    nibabel>=5.0 \
    numpy>=1.24 \
    scipy>=1.11 \
    matplotlib>=3.7 \
    pyyaml>=6.0 \
    pillow>=10.0 \
    pydantic>=2.0

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Model checkpoint is mounted at runtime (not baked into image):
#   -v ./outputs:/app/outputs
# Default checkpoint path: /app/outputs/fold_0/checkpoints/best_model.pth

# ---------------------------------------------------------------------------
# FastAPI target
# ---------------------------------------------------------------------------
FROM base AS api

RUN pip install --no-cache-dir \
    fastapi>=0.104 \
    uvicorn[standard]>=0.24 \
    python-multipart>=0.0.6

EXPOSE 8000

ENV DEVICE=cpu
ENV MODEL_NAME=attention_unet3d
ENV CHECKPOINT_PATH=/app/outputs/fold_0/checkpoints/best_model.pth

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Gradio demo target
# ---------------------------------------------------------------------------
FROM base AS demo

RUN pip install --no-cache-dir gradio>=4.0

COPY demo/ demo/

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "demo/app.py"]
