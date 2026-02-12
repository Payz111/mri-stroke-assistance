#!/bin/bash
# Download ISLES 2024 dataset
# Check https://isles-24.grand-challenge.org/ for latest download instructions

set -euo pipefail

DATA_DIR="data/raw/isles24"
mkdir -p "$DATA_DIR"

echo "Downloading ISLES 2024 dataset..."
echo ""
echo "NOTE: ISLES 2024 may require registration."
echo "Please check: https://isles-24.grand-challenge.org/"
echo ""
echo "If using HuggingFace:"
echo "  pip install huggingface_hub"
echo "  huggingface-cli download stroke-segmentation/isles24-stroke --local-dir $DATA_DIR"
echo ""

echo "Please download the dataset manually and place it in: $DATA_DIR"
