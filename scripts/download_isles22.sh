#!/bin/bash
# Download ISLES 2022 dataset from Zenodo
# https://zenodo.org/record/7153326

set -euo pipefail

DATA_DIR="data/raw/isles22"
mkdir -p "$DATA_DIR"

echo "Downloading ISLES 2022 dataset..."
echo "Source: https://zenodo.org/record/7153326"
echo ""
echo "NOTE: If automatic download fails, please download manually from:"
echo "  https://zenodo.org/record/7153326"
echo "and extract to: $DATA_DIR"
echo ""

# Download
wget -c https://zenodo.org/record/7153326/files/ISLES-2022.zip -P "$DATA_DIR"

# Extract
echo "Extracting..."
unzip "$DATA_DIR/ISLES-2022.zip" -d "$DATA_DIR"

echo "Done. Data saved to $DATA_DIR"
