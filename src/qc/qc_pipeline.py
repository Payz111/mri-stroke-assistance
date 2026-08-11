"""Quality-control gates for preprocessed MRI data.

Implements a battery of automated checks that must pass before a subject
is accepted into the training / inference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QCResult:
    """Outcome of a single QC check."""

    passed: bool
    reason: str


class QCPipeline:
    """Run all quality-control checks for a single subject.

    Each ``check_*`` method returns a :class:`QCResult`.  Call
    :meth:`run_all` to execute every gate and get a combined report.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise with an optional configuration dictionary.

        Parameters
        ----------
        config:
            Thresholds and tolerances for each QC check.
        """
        raise NotImplementedError()

    def check_modalities(self, subject_data: dict[str, Any]) -> QCResult:
        """Verify that all required modalities (DWI, ADC, FLAIR) are present.

        Parameters
        ----------
        subject_data:
            Dictionary keyed by modality name, values are volumes or paths.

        Returns
        -------
        QCResult
        """
        raise NotImplementedError()

    def check_spacing(self, subject_data: dict[str, Any]) -> QCResult:
        """Verify that voxel spacing is within acceptable bounds.

        Parameters
        ----------
        subject_data:
            Must include ``spacing`` metadata per modality.

        Returns
        -------
        QCResult
        """
        raise NotImplementedError()

    def check_coverage(self, subject_data: dict[str, Any]) -> QCResult:
        """Verify sufficient brain coverage (field-of-view check).

        Parameters
        ----------
        subject_data:
            Volume arrays and associated metadata.

        Returns
        -------
        QCResult
        """
        raise NotImplementedError()

    def check_dwi_adc_consistency(self, subject_data: dict[str, Any]) -> QCResult:
        """Check that DWI and ADC maps are inversely consistent.

        Lesion regions bright on DWI should be dark on ADC and vice-versa.

        Parameters
        ----------
        subject_data:
            Must contain ``dwi`` and ``adc`` arrays plus an optional mask.

        Returns
        -------
        QCResult
        """
        raise NotImplementedError()

    def run_all(self, subject_data: dict[str, Any]) -> list[QCResult]:
        """Execute every QC gate and return the list of results.

        Parameters
        ----------
        subject_data:
            Full subject dictionary containing all modalities and metadata.

        Returns
        -------
        list[QCResult]
            One :class:`QCResult` per check, in the order they were run.
        """
        raise NotImplementedError()
