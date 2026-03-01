FROM python:3.11-slim

WORKDIR /app

# System dependencies for scipy, matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller image)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Python dependencies (minimal for demo)
RUN pip install --no-cache-dir \
    monai>=1.3 \
    nibabel>=5.0 \
    numpy>=1.24 \
    scipy>=1.11 \
    matplotlib>=3.7 \
    pyyaml>=6.0 \
    gradio>=4.0 \
    pillow>=10.0

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/
COPY demo/ demo/

# Copy checkpoint if available (optional at build time)
# Users can mount it at runtime: -v ./outputs:/app/outputs
COPY outputs/fold_0/checkpoints/ outputs/fold_0/checkpoints/

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "demo/app.py"]
