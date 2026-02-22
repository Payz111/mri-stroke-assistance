"""Co-registration of multi-modal MRI volumes.

Provides rigid registration so that all modalities share the same
coordinate frame (DWI space) before segmentation.

Uses SimpleITK for registration. For production use, ANTsPy can be
enabled via the 'ants' optional dependency.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def coregister(
    moving: np.ndarray,
    fixed: np.ndarray,
    moving_spacing: tuple[float, ...] | None = None,
    fixed_spacing: tuple[float, ...] | None = None,
    method: str = "rigid",
) -> np.ndarray:
    """Register *moving* volume to *fixed* reference space.

    Parameters
    ----------
    moving:
        The volume to be transformed (e.g. FLAIR).
    fixed:
        The reference volume (e.g. DWI).
    moving_spacing:
        Voxel spacing of the moving image (x, y, z).
    fixed_spacing:
        Voxel spacing of the fixed image (x, y, z).
    method:
        Registration strategy -- ``"rigid"`` or ``"affine"``.

    Returns
    -------
    np.ndarray
        The *moving* volume resampled into *fixed* space.
    """
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

    if fixed_spacing is not None:
        fixed_sitk.SetSpacing([float(s) for s in fixed_spacing])
    if moving_spacing is not None:
        moving_sitk.SetSpacing([float(s) for s in moving_spacing])

    # Set up registration
    reg = sitk.ImageRegistrationMethod()

    # Metric: mutual information (robust for multi-modal MRI)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)

    # Interpolator
    reg.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Transform
    if method == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk,
            moving_sitk,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    elif method == "affine":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk,
            moving_sitk,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'rigid' or 'affine'.")

    reg.SetInitialTransform(initial_transform, inPlace=False)

    # Run
    final_transform = reg.Execute(fixed_sitk, moving_sitk)

    # Resample moving into fixed space
    resampled = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_sitk.GetPixelID(),
    )

    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def resample_to_reference(
    moving: np.ndarray,
    fixed: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Resample *moving* to match the spatial shape of *fixed*.

    A simpler alternative to full registration when volumes are already
    roughly aligned (e.g. same patient, same session).

    Parameters
    ----------
    order:
        Interpolation order (0=nearest, 1=linear, 3=cubic).
    """
    from scipy.ndimage import zoom

    if moving.shape == fixed.shape:
        return moving

    factors = [r / s for r, s in zip(fixed.shape, moving.shape)]
    return zoom(moving, factors, order=order).astype(np.float32)
