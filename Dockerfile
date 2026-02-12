FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY demo/ demo/
COPY scripts/ scripts/

# Expose Gradio port
EXPOSE 7860

# Default: run demo
CMD ["python", "demo/app.py"]
