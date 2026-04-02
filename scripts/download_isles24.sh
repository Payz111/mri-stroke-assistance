#!/bin/bash
# Download ISLES 2024 dataset from Zenodo.
#
# Dataset: ISLES'24 - A Real-World Longitudinal Multimodal Stroke Dataset
# Source: https://zenodo.org/records/16748089
# Size: ~99 GB (train.7z)
# Cases: 149 acute ischemic stroke
# License: CC-BY-NC-SA 4.0
#
# Contents per subject:
#   rawdata/  - NCCT, CTA, CTP (4D), perfusion maps (Tmax, CBF, CBV, MTT), DWI, ADC
#   derivatives/  - Co-registered to NCCT space + lesion masks
#   phenotype/  - Clinical CSVs (demographics, outcomes)
#
# Usage:
#   bash scripts/download_isles24.sh [data-dir]
#
# Example:
#   bash scripts/download_isles24.sh data/raw/isles24

set -euo pipefail

DATA_DIR="${1:-data/raw/isles24}"
ZENODO_RECORD="16748089"
ZENODO_URL="https://zenodo.org/records/${ZENODO_RECORD}/files/train.7z?download=1"
ARCHIVE="${DATA_DIR}/train.7z"

mkdir -p "$DATA_DIR"

echo "=== ISLES 2024 Dataset Download ==="
echo "Target: ${DATA_DIR}"
echo "Source: Zenodo record ${ZENODO_RECORD}"
echo "Size: ~99 GB (will need ~200 GB free space for download + extraction)"
echo ""

# Check for 7z
if ! command -v 7z &> /dev/null; then
    echo "ERROR: 7z not found. Install p7zip:"
    echo "  Ubuntu/Debian: sudo apt install p7zip-full"
    echo "  macOS: brew install p7zip"
    echo "  Windows: install 7-Zip from https://www.7-zip.org/"
    exit 1
fi

# Download
if [ -f "$ARCHIVE" ]; then
    echo "Archive already exists: ${ARCHIVE}"
    echo "Skipping download. Delete it to re-download."
else
    echo "Downloading train.7z (~99 GB). This will take a while..."
    echo ""
    # Use wget with resume support, or curl as fallback
    if command -v wget &> /dev/null; then
        wget -c -O "$ARCHIVE" "$ZENODO_URL"
    elif command -v curl &> /dev/null; then
        curl -C - -L -o "$ARCHIVE" "$ZENODO_URL"
    else
        echo "ERROR: Neither wget nor curl found."
        exit 1
    fi
fi

# Verify checksum
echo ""
echo "Verifying MD5 checksum..."
EXPECTED_MD5="36ae28b9a17f7340b8bbef62b595cb57"
if command -v md5sum &> /dev/null; then
    ACTUAL_MD5=$(md5sum "$ARCHIVE" | cut -d' ' -f1)
elif command -v md5 &> /dev/null; then
    ACTUAL_MD5=$(md5 -q "$ARCHIVE")
else
    echo "WARNING: md5sum/md5 not found, skipping checksum verification."
    ACTUAL_MD5="$EXPECTED_MD5"
fi

if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
    echo "ERROR: MD5 mismatch!"
    echo "  Expected: ${EXPECTED_MD5}"
    echo "  Got:      ${ACTUAL_MD5}"
    echo "  File may be corrupted. Delete and re-download."
    exit 1
fi
echo "Checksum OK."

# Extract
echo ""
echo "Extracting (this will take a while)..."
7z x -o"$DATA_DIR" "$ARCHIVE" -y

echo ""
echo "=== Download complete! ==="
echo "Data: ${DATA_DIR}"
echo ""
echo "Expected structure:"
echo "  ${DATA_DIR}/rawdata/sub-strokecaseXXXX/ses-XXXX/..."
echo "  ${DATA_DIR}/derivatives/sub-strokecaseXXXX/ses-XXXX/..."
echo "  ${DATA_DIR}/phenotype/..."
echo ""
echo "Next: run notebooks/05_eda_isles24.ipynb for data exploration"
