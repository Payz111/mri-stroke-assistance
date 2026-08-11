"""Vascular-territory classification.

Determines which arterial territory a lesion most likely belongs to
(e.g. MCA, ACA, PCA, vertebrobasilar) using a heuristic based on
lesion centroid position within the normalised volume.
"""

from __future__ import annotations

import numpy as np


def classify_vascular_territory(
    mask: np.ndarray,
    atlas: np.ndarray | None = None,
) -> tuple[str, float]:
    """Classify the dominant vascular territory of the lesion.

    When no atlas is provided, uses a heuristic mapping:
    - Superior + anterior/lateral -> MCA (most common ~70% of strokes)
    - Superior + medial/frontal -> ACA
    - Posterior + superior -> PCA
    - Inferior (below z=0.25) -> vertebrobasilar
    - Border zones -> watershed

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask.
    atlas:
        Vascular-territory atlas volume. If ``None``, the heuristic is used.

    Returns
    -------
    tuple[str, float]
        ``(territory_name, confidence)`` where *confidence* is
        between 0 and 1. Lower confidence for ambiguous locations.
    """
    if mask.sum() == 0:
        return ("uncertain", 0.0)

    if atlas is not None:
        return _atlas_based_territory(mask, atlas)

    return _heuristic_territory(mask)


def _atlas_based_territory(mask: np.ndarray, atlas: np.ndarray) -> tuple[str, float]:
    """Classify using an integer-labelled vascular territory atlas."""
    labels = atlas[mask > 0]
    unique, counts = np.unique(labels, return_counts=True)

    valid = unique > 0
    unique = unique[valid]
    counts = counts[valid]

    if len(unique) == 0:
        return ("uncertain", 0.0)

    dominant_idx = np.argmax(counts)
    confidence = float(counts[dominant_idx] / counts.sum())
    return (str(int(unique[dominant_idx])), confidence)


def _heuristic_territory(mask: np.ndarray) -> tuple[str, float]:
    """Estimate vascular territory from centroid position."""
    coords = np.argwhere(mask > 0)
    centroid = coords.mean(axis=0)
    shape = np.array(mask.shape, dtype=float)
    norm = centroid / shape  # [0..1]

    z_frac = norm[2] if len(norm) >= 3 else 0.5
    lr_frac = norm[0]  # left-right
    ap_frac = norm[1]  # anterior-posterior

    # Distance from midline (0.5 = midline)
    midline_dist = abs(lr_frac - 0.5)

    # Infratentorial -> vertebrobasilar
    if z_frac < 0.25:
        return ("vertebrobasilar", 0.75)

    # Deep central structures near midline -> could be MCA perforators or ACA
    is_deep = min(lr_frac, 1 - lr_frac, ap_frac, 1 - ap_frac) > 0.35

    # Superior regions
    if z_frac >= 0.25:
        # Medial frontal -> ACA
        if midline_dist < 0.15 and ap_frac > 0.55:
            return ("ACA", 0.60)

        # Posterior -> PCA
        if ap_frac < 0.35 and z_frac < 0.65:
            return ("PCA", 0.60)

        # Lateral -> MCA (most common)
        if midline_dist > 0.15:
            return ("MCA", 0.70)

        # Deep central -> MCA perforators (lenticulostriates)
        if is_deep:
            return ("MCA", 0.55)

    # Border zone between territories
    return ("uncertain", 0.30)
