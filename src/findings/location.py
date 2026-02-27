"""Anatomic-location mapping of lesion masks.

Maps segmented lesion voxels to named brain regions using a heuristic
based on the lesion centroid position within the normalised volume.
Without a registered atlas, this provides a rough but deterministic estimate.
"""
from __future__ import annotations

import numpy as np


# Relative depth zones along the Z axis (inferior -> superior)
# Values are fractions of total depth [0..1]
_Z_ZONES = {
    "cerebellum": (0.0, 0.20),
    "brainstem": (0.05, 0.25),
    "temporal": (0.15, 0.50),
    "occipital": (0.15, 0.55),
    "basal_ganglia": (0.30, 0.55),
    "thalamus": (0.30, 0.50),
    "internal_capsule": (0.30, 0.55),
    "insula": (0.35, 0.60),
    "parietal": (0.50, 0.85),
    "frontal": (0.55, 1.0),
}


def map_anatomic_location(
    mask: np.ndarray,
    atlas: np.ndarray | None = None,
) -> list[str]:
    """Identify the anatomic brain regions overlapped by the lesion.

    When no atlas is provided, uses a heuristic based on the lesion
    centroid's normalised position within the volume. This is a rough
    approximation sufficient for structured reporting.

    Parameters
    ----------
    mask:
        Binary 3-D lesion mask (shape D, H, W or similar).
    atlas:
        Integer-labelled atlas volume (same shape as *mask*).
        If ``None``, the centroid heuristic is used.

    Returns
    -------
    list[str]
        Region names (from AnatomicLocation enum values).
        Returns ``["multiple"]`` if the lesion is very large (>10% of brain).
    """
    if mask.sum() == 0:
        return []

    if atlas is not None:
        return _atlas_based_location(mask, atlas)

    return _heuristic_location(mask)


def _atlas_based_location(mask: np.ndarray, atlas: np.ndarray) -> list[str]:
    """Map regions using an integer-labelled atlas."""
    lesion_labels = atlas[mask > 0]
    unique, counts = np.unique(lesion_labels, return_counts=True)

    # Filter out background (label 0)
    valid = unique > 0
    unique = unique[valid]
    counts = counts[valid]

    if len(unique) == 0:
        return ["multiple"]

    # Sort by descending count
    order = np.argsort(-counts)
    return [str(int(u)) for u in unique[order]]


def _heuristic_location(mask: np.ndarray) -> list[str]:
    """Estimate location from centroid position in normalised coordinates."""
    coords = np.argwhere(mask > 0)
    centroid = coords.mean(axis=0)

    shape = np.array(mask.shape, dtype=float)
    norm = centroid / shape  # normalised [0..1] for each axis

    # Volume fraction -- very large lesion spans multiple regions
    brain_voxels = max(np.prod(shape) * 0.4, 1)  # ~40% of volume is brain
    lesion_frac = mask.sum() / brain_voxels
    if lesion_frac > 0.10:
        return ["multiple"]

    # axis 0 = Left-Right, axis 1 = Anterior-Posterior, axis 2 = Inferior-Superior
    # In many NIfTI orientations: dim0=sagittal, dim1=coronal, dim2=axial
    # We use dim2 as the Z (axial) axis for depth estimation
    z_frac = norm[2] if len(norm) >= 3 else 0.5

    # Check depth (how deep from surface = distance from edges on X/Y)
    xy_center_dist = min(norm[0], 1 - norm[0], norm[1], 1 - norm[1])
    is_deep = xy_center_dist > 0.35  # close to centre = deep structure

    regions = []

    # Deep structures
    if is_deep and 0.30 <= z_frac <= 0.55:
        regions.append("basal_ganglia")
    elif is_deep and 0.30 <= z_frac <= 0.50:
        regions.append("thalamus")

    # Posterior-inferior: cerebellum / brainstem
    if z_frac < 0.20:
        if is_deep:
            regions.append("brainstem")
        else:
            regions.append("cerebellum")

    # Cortical zones by Z position
    if z_frac >= 0.55:
        if norm[1] > 0.55:  # anterior
            regions.append("frontal")
        else:
            regions.append("parietal")
    elif 0.15 <= z_frac < 0.55:
        if norm[1] > 0.6:  # anterior-lateral
            regions.append("temporal")
        elif norm[1] < 0.35:  # posterior
            regions.append("occipital")

    if not regions:
        regions.append("multiple")

    return regions
