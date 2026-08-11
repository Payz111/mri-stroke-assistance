"""CTP perfusion visualization.

Creates overlays showing core (red) and penumbra (green) on NCCT/Tmax.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def create_perfusion_overlay(
    background: np.ndarray,
    core_mask: np.ndarray,
    penumbra_mask: np.ndarray,
    slice_idx: int | None = None,
    core_color: tuple[int, int, int] = (255, 50, 50),
    penumbra_color: tuple[int, int, int] = (50, 255, 50),
    alpha: float = 0.4,
) -> Image.Image:
    """Create a single slice with core and penumbra overlay.

    Parameters
    ----------
    background:
        3D volume for background (NCCT or Tmax).
    core_mask:
        Binary core mask (D, H, W).
    penumbra_mask:
        Binary penumbra mask (D, H, W).
    slice_idx:
        Axial slice index. If None, picks slice with most pathology.
    core_color:
        RGB color for core overlay (default red).
    penumbra_color:
        RGB color for penumbra overlay (default green).
    alpha:
        Overlay transparency (0-1).

    Returns
    -------
    PIL.Image
        RGB image with overlay.
    """
    if slice_idx is None:
        # Pick slice with most core + penumbra voxels
        combined = (core_mask > 0).astype(float) + (penumbra_mask > 0).astype(float)
        per_slice = combined.sum(axis=(0, 1))
        slice_idx = int(np.argmax(per_slice))

    bg_slice = background[:, :, slice_idx].astype(np.float32)
    core_slice = core_mask[:, :, slice_idx] > 0
    penumbra_slice = penumbra_mask[:, :, slice_idx] > 0

    # Normalize background to [0, 255]
    bg_min, bg_max = (
        np.percentile(bg_slice[bg_slice > 0], [1, 99]) if (bg_slice > 0).any() else (0, 1)
    )
    bg_norm = np.clip((bg_slice - bg_min) / max(bg_max - bg_min, 1e-7) * 255, 0, 255)
    bg_uint8 = bg_norm.astype(np.uint8)

    # Create RGB
    rgb = np.stack([bg_uint8, bg_uint8, bg_uint8], axis=-1)

    # Apply penumbra overlay (green) first, then core (red) on top
    for ch in range(3):
        rgb[penumbra_slice, ch] = (
            (1 - alpha) * rgb[penumbra_slice, ch] + alpha * penumbra_color[ch]
        ).astype(np.uint8)
        rgb[core_slice, ch] = ((1 - alpha) * rgb[core_slice, ch] + alpha * core_color[ch]).astype(
            np.uint8
        )

    return Image.fromarray(rgb)


def create_perfusion_montage(
    background: np.ndarray,
    core_mask: np.ndarray,
    penumbra_mask: np.ndarray,
    tmax: np.ndarray | None = None,
    n_slices: int = 6,
) -> Image.Image:
    """Create a montage showing perfusion analysis across multiple slices.

    Two rows:
    - Top: Background (NCCT) with core (red) + penumbra (green) overlay
    - Bottom: Tmax colormap (if provided) or repeat of overlay

    Parameters
    ----------
    background:
        3D volume (NCCT or similar).
    core_mask:
        Binary core mask.
    penumbra_mask:
        Binary penumbra mask.
    tmax:
        Optional Tmax map for second row.
    n_slices:
        Number of slices in the montage.

    Returns
    -------
    PIL.Image
        Montage image.
    """
    import matplotlib.pyplot as plt

    # Find slices with pathology
    combined = (core_mask > 0).astype(float) + (penumbra_mask > 0).astype(float)
    per_slice = combined.sum(axis=(0, 1))

    if per_slice.max() > 0:
        # Pick slices with most pathology, evenly spaced
        nonzero = np.where(per_slice > 0)[0]
        if len(nonzero) >= n_slices:
            indices = np.linspace(0, len(nonzero) - 1, n_slices, dtype=int)
            slice_indices = nonzero[indices]
        else:
            # Pad with neighboring slices
            center = nonzero[len(nonzero) // 2]
            half = n_slices // 2
            start = max(0, center - half)
            end = min(background.shape[2], start + n_slices)
            slice_indices = np.arange(start, end)
    else:
        # No pathology found, show middle slices
        mid = background.shape[2] // 2
        half = n_slices // 2
        slice_indices = np.arange(max(0, mid - half), min(background.shape[2], mid + half + 1))[
            :n_slices
        ]

    n_rows = 2 if tmax is not None else 1
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(3 * n_slices, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for col, s_idx in enumerate(slice_indices):
        ax = axes[0][col] if n_slices > 1 else axes[0]

        # Row 1: NCCT + overlay
        bg = background[:, :, s_idx].astype(np.float32)
        bg_min, bg_max = np.percentile(bg[bg > 0], [1, 99]) if (bg > 0).any() else (0, 1)
        bg_norm = np.clip((bg - bg_min) / max(bg_max - bg_min, 1e-7), 0, 1)

        ax.imshow(bg_norm.T, cmap="gray", origin="lower")

        # Overlay masks
        core_s = core_mask[:, :, s_idx] > 0
        penumbra_s = penumbra_mask[:, :, s_idx] > 0

        if penumbra_s.any():
            overlay_p = np.zeros((*penumbra_s.shape, 4))
            overlay_p[penumbra_s] = [0, 1, 0, 0.4]  # green
            ax.imshow(overlay_p.transpose(1, 0, 2), origin="lower")
        if core_s.any():
            overlay_c = np.zeros((*core_s.shape, 4))
            overlay_c[core_s] = [1, 0, 0, 0.5]  # red
            ax.imshow(overlay_c.transpose(1, 0, 2), origin="lower")

        ax.set_title(f"z={s_idx}", fontsize=8)
        ax.axis("off")

        # Row 2: Tmax colormap
        if tmax is not None and n_rows > 1:
            ax2 = axes[1][col] if n_slices > 1 else axes[1]
            tmax_s = tmax[:, :, s_idx].astype(np.float32)
            ax2.imshow(
                tmax_s.T,
                cmap="hot",
                origin="lower",
                vmin=0,
                vmax=20,
            )
            ax2.set_title(f"Tmax z={s_idx}", fontsize=8)
            ax2.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    return Image.fromarray(buf)
