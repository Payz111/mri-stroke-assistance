"""HuggingFace Spaces entry point for MRI Stroke Assist demo.

This wrapper sets up paths and launches the Gradio demo.
The checkpoint (best_model.pth) should be in the Space repo
at outputs/fold_0/checkpoints/best_model.pth.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root is the Space repo root (where this file lives)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Patch the demo module's checkpoint path to use Space-relative path
import demo.app as demo_app

CHECKPOINT = ROOT / "outputs" / "fold_0" / "checkpoints" / "best_model.pth"
demo_app.DEFAULT_CHECKPOINT = CHECKPOINT

if CHECKPOINT.exists():
    print(f"Checkpoint found: {CHECKPOINT}")
else:
    print(
        f"Checkpoint not found at {CHECKPOINT}. "
        "Only Synthetic Demo will work. "
        "Upload best_model.pth to outputs/fold_0/checkpoints/ for real inference."
    )

# Launch the app
app = demo_app.create_app()
app.launch(server_name="0.0.0.0", server_port=7860)
