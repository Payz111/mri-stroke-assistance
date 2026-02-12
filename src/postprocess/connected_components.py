"""Connected-component analysis for lesion-mask refinement."""
from __future__ import annotations

import numpy as np


def extract_connected_components(
    mask: np.ndarray,
    min_size: int = 50,
) -> tuple[np.ndarray, int]:
    """Label connected components and remove those below *min_size*.

    Parameters
    ----------
    mask:
        Binary 3-D mask (dtype ``uint8`` or ``bool``).
    min_size:
        Minimum number of voxels for a component to be retained.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(labelled_mask, num_components)`` — the labelled array where each
        retained component has a unique integer ID, and the count of
        retained components.
    """
    raise NotImplementedError()
