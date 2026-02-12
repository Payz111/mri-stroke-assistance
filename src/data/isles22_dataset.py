"""PyTorch Dataset for the ISLES'22 ischemic-stroke segmentation challenge.

Loads DWI, ADC, FLAIR volumes and the corresponding lesion mask for each
subject, returning them as a dictionary suitable for 3-D segmentation
training pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ISLES22Dataset(Dataset):
    """Dataset class that serves ISLES'22 subjects.

    Each item is a dictionary with the keys:
        - ``dwi``      : DWI volume tensor
        - ``adc``      : ADC volume tensor
        - ``flair``    : FLAIR volume tensor
        - ``mask``     : Ground-truth lesion mask tensor
        - ``metadata`` : dict of per-subject metadata (spacing, affine, …)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Any | None = None,
    ) -> None:
        """Initialise the dataset.

        Parameters
        ----------
        root_dir:
            Root directory that contains the ISLES'22 data
            (expected sub-folders per subject).
        split:
            One of ``"train"``, ``"val"``, or ``"test"``.
        transform:
            Optional callable (e.g. a MONAI ``Compose``) applied to the
            sample dictionary before it is returned.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return the number of subjects in this split."""
        raise NotImplementedError()

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return the sample at *index*.

        Returns
        -------
        dict
            Keys: ``dwi``, ``adc``, ``flair``, ``mask``, ``metadata``.
        """
        raise NotImplementedError()
