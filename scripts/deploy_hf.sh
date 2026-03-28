#!/usr/bin/env bash
# Deploy MRI Stroke Assist demo to HuggingFace Spaces.
#
# Prerequisites:
#   1. Create a HuggingFace account at https://huggingface.co/join
#   2. Install huggingface_hub: pip install huggingface_hub
#   3. Login: huggingface-cli login
#   4. Create a Space at https://huggingface.co/new-space
#      - Name: mri-stroke-assist
#      - SDK: Gradio
#      - Hardware: CPU Basic (free)
#
# Usage:
#   bash scripts/deploy_hf.sh <your-hf-username> [checkpoint-path]
#
# Example:
#   bash scripts/deploy_hf.sh myusername outputs/fold_0/checkpoints/best_model.pth

set -euo pipefail

USERNAME="${1:?Usage: deploy_hf.sh <hf-username> [checkpoint-path]}"
CHECKPOINT="${2:-}"
SPACE_NAME="mri-stroke-assist"
SPACE_REPO="https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="${PROJECT_ROOT}/deploy/hf_spaces"
TEMP_DIR="$(mktemp -d)"

echo "=== MRI Stroke Assist - HuggingFace Spaces Deployment ==="
echo "Space: ${SPACE_REPO}"
echo "Temp dir: ${TEMP_DIR}"

# Clone the Space repo (or create if empty)
echo ""
echo ">>> Cloning Space repo..."
git clone "${SPACE_REPO}" "${TEMP_DIR}/space" 2>/dev/null || {
    echo "Could not clone. Make sure the Space exists at ${SPACE_REPO}"
    echo "Create it at: https://huggingface.co/new-space"
    rm -rf "${TEMP_DIR}"
    exit 1
}

cd "${TEMP_DIR}/space"

# Copy HF Spaces config files
echo ">>> Copying deployment files..."
cp "${DEPLOY_DIR}/app.py" .
cp "${DEPLOY_DIR}/requirements.txt" .
cp "${DEPLOY_DIR}/README.md" .

# Copy source code
echo ">>> Copying source code..."
cp -r "${PROJECT_ROOT}/src" .
cp -r "${PROJECT_ROOT}/configs" .
cp -r "${PROJECT_ROOT}/demo" .

# Copy checkpoint if provided
if [ -n "${CHECKPOINT}" ] && [ -f "${PROJECT_ROOT}/${CHECKPOINT}" ]; then
    echo ">>> Copying checkpoint: ${CHECKPOINT}"
    mkdir -p outputs/fold_0/checkpoints
    cp "${PROJECT_ROOT}/${CHECKPOINT}" outputs/fold_0/checkpoints/best_model.pth
elif [ -n "${CHECKPOINT}" ]; then
    echo "WARNING: Checkpoint not found at ${PROJECT_ROOT}/${CHECKPOINT}"
    echo "Only Synthetic Demo will work."
fi

# Git add and push
echo ""
echo ">>> Pushing to HuggingFace Spaces..."
git add -A
git commit -m "Deploy MRI Stroke Assist demo" || echo "No changes to commit"
git push

echo ""
echo "=== Deployment complete! ==="
echo "Space URL: ${SPACE_REPO}"
echo ""
echo "The Space will build automatically. Check the logs at:"
echo "${SPACE_REPO}/logs"

# Cleanup
cd /
rm -rf "${TEMP_DIR}"
