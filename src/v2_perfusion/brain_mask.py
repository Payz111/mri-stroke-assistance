"""Brain mask estimation from CT images."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, label


def estimate_brain_mask_ct(
    ncct: np.ndarray,
    low_hu: float = 0.0,
    high_hu: float = 100.0,
) -> np.ndarray:
    """Estimate a brain mask from NCCT using Hounsfield unit thresholding.

    Brain tissue: ~20-45 HU, CSF: ~0-15 HU, bone: >500 HU.
    Uses a generous range [low_hu, high_hu] then keeps the largest
    connected component and fills holes.

    Parameters
    ----------
    ncct:
        NCCT volume (D, H, W) in Hounsfield units.
    low_hu:
        Lower HU threshold (default 0).
    high_hu:
        Upper HU threshold (default 100).

    Returns
    -------
    np.ndarray
        Binary brain mask (D, H, W) float32.
    """
    mask = ((ncct >= low_hu) & (ncct <= high_hu)).astype(np.float32)

    # Keep largest connected component
    labelled, n_components = label(mask)
    if n_components > 1:
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0  # ignore background
        largest = sizes.argmax()
        mask = (labelled == largest).astype(np.float32)

    # Fill holes slice-by-slice (axial)
    for z in range(mask.shape[2]):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z]).astype(np.float32)

    return mask
