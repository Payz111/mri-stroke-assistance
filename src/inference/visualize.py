"""Visualization utilities for inference results.

Creates overlay images showing DWI slices with predicted lesion masks.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np


def _normalize_slice(img_slice: np.ndarray) -> np.ndarray:
    """Normalize a 2-D slice to [0, 1] for display."""
    vmin = np.percentile(img_slice, 1)
    vmax = np.percentile(img_slice, 99)
    if vmax - vmin < 1e-7:
        return np.zeros_like(img_slice, dtype=np.float32)
    return np.clip((img_slice - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def find_best_slice(mask: np.ndarray) -> int:
    """Find the axial slice with the most lesion voxels."""
    if mask.ndim != 3 or mask.sum() == 0:
        return mask.shape[2] // 2 if mask.ndim == 3 else 0
    per_slice = (mask > 0).sum(axis=(0, 1))
    return int(np.argmax(per_slice))


def create_overlay_image(
    dwi: np.ndarray,
    pred_mask: np.ndarray,
    slice_idx: int | None = None,
) -> Any:
    """Create a DWI slice with red lesion overlay.

    Parameters
    ----------
    dwi:
        DWI volume (D, H, W).
    pred_mask:
        Predicted binary mask (D, H, W).
    slice_idx:
        Axial slice index. If None, picks slice with most lesion voxels.

    Returns
    -------
    PIL.Image.Image
        RGB image with overlay.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return None

    if slice_idx is None:
        slice_idx = find_best_slice(pred_mask)

    # Get axial slices (axis 2)
    dwi_slice = dwi[:, :, slice_idx].T  # Transpose for standard orientation
    mask_slice = pred_mask[:, :, slice_idx].T

    dwi_norm = _normalize_slice(dwi_slice)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.imshow(dwi_norm, cmap="gray", aspect="equal")

    # Red overlay for lesion
    if mask_slice.sum() > 0:
        overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        overlay[mask_slice > 0] = [1, 0, 0, 0.4]  # Red with 40% opacity
        ax.imshow(overlay, aspect="equal")

        # Contour
        ax.contour(mask_slice > 0, levels=[0.5], colors=["red"], linewidths=1.5)

    n_lesion_voxels = int(pred_mask.sum())
    ax.set_title(f"Axial slice {slice_idx} | Lesion voxels: {n_lesion_voxels}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def create_montage(
    dwi: np.ndarray,
    pred_mask: np.ndarray,
    n_slices: int = 6,
) -> Any:
    """Create a montage of axial slices with lesion overlay.

    Parameters
    ----------
    dwi:
        DWI volume (D, H, W).
    pred_mask:
        Predicted binary mask (D, H, W).
    n_slices:
        Number of slices to show.

    Returns
    -------
    PIL.Image.Image
        RGB montage image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return None

    n_axial = dwi.shape[2]

    if pred_mask.sum() > 0:
        # Pick slices around the lesion
        per_slice = (pred_mask > 0).sum(axis=(0, 1))
        best = int(np.argmax(per_slice))
        spread = max(n_axial // (n_slices + 2), 1)
        indices = []
        for offset in range(-(n_slices // 2), n_slices // 2 + 1):
            idx = best + offset * spread
            if 0 <= idx < n_axial:
                indices.append(idx)
        indices = sorted(set(indices))[:n_slices]
    else:
        # Evenly spaced slices
        indices = np.linspace(0, n_axial - 1, n_slices, dtype=int).tolist()

    ncols = min(n_slices, 3)
    nrows = (len(indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=80)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, ax in enumerate(axes.flat):
        if i < len(indices):
            idx = indices[i]
            dwi_slice = dwi[:, :, idx].T
            mask_slice = pred_mask[:, :, idx].T
            dwi_norm = _normalize_slice(dwi_slice)

            ax.imshow(dwi_norm, cmap="gray", aspect="equal")
            if mask_slice.sum() > 0:
                overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
                overlay[mask_slice > 0] = [1, 0, 0, 0.4]
                ax.imshow(overlay, aspect="equal")
                ax.contour(mask_slice > 0, levels=[0.5], colors=["red"], linewidths=1)
            ax.set_title(f"Slice {idx}", fontsize=9, color="white")
        ax.axis("off")

    fig.patch.set_facecolor("black")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img
