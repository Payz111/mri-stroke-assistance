"""False-positive filtering using DWI/ADC intensity priors.

Removes spurious lesion detections that do not exhibit the expected
DWI-bright / ADC-dark signal pattern of acute ischaemia.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def filter_false_positives(
    mask: np.ndarray,
    dwi: np.ndarray,
    adc: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Remove false-positive lesion components from *mask*.

    Parameters
    ----------
    mask:
        Binary or labelled 3-D lesion mask.
    dwi:
        Co-registered DWI volume.
    adc:
        Co-registered ADC volume.
    config:
        Optional thresholds for DWI brightness and ADC darkness checks.

    Returns
    -------
    np.ndarray
        Cleaned binary mask with the same shape as *mask*.
    """
    raise NotImplementedError()
