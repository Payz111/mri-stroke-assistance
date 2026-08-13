"""Quality-control gates for preprocessed MRI data.

Implements a battery of automated checks that must pass before a subject
is accepted into the training / inference pipeline.

Policy (ADR-006): a *critical* failure means no prediction is produced at all.
In a clinical setting a confident wrong answer is more dangerous than no
answer, so the pipeline refuses rather than guessing on unreadable input.
Checks marked *warning* are reported but do not block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CRITICAL = "critical"
WARNING = "warning"

DEFAULT_CONFIG: dict[str, Any] = {
    # Modalities the V1 pipeline cannot run without.
    "required_modalities": ("dwi", "adc", "flair"),
    # Plausible voxel sizes for brain MRI, in mm. ISLES 2022 spans 0.88-2.0
    # in-plane and 2.0-5.0 through-plane; the bounds are deliberately wider.
    "min_spacing_mm": 0.3,
    "max_spacing_mm": 6.0,
    # Smallest volume that can still carry a readable brain.
    "min_shape": (32, 32, 8),
    # Fraction of non-zero voxels expected inside a head-containing volume.
    "min_brain_fraction": 0.02,
    # Above this correlation, DWI and ADC are effectively the same image --
    # almost always the same file passed twice.
    "max_dwi_adc_correlation": 0.98,
}


@dataclass
class QCResult:
    """Outcome of a single QC check."""

    passed: bool
    reason: str
    name: str = ""
    severity: str = CRITICAL


@dataclass
class QCReport:
    """Combined outcome of every gate for one subject."""

    passed: bool
    results: list[QCResult] = field(default_factory=list)

    @property
    def critical_failures(self) -> list[QCResult]:
        return [r for r in self.results if not r.passed and r.severity == CRITICAL]

    @property
    def warnings(self) -> list[QCResult]:
        return [r for r in self.results if not r.passed and r.severity == WARNING]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for the findings payload and the API."""
        return {
            "passed": self.passed,
            "checks": {
                r.name: {"passed": r.passed, "severity": r.severity, "reason": r.reason}
                for r in self.results
            },
            "critical_failures": [r.reason for r in self.critical_failures],
            "warnings": [r.reason for r in self.warnings],
        }

    def summary_text(self) -> str:
        """Human-readable refusal text used in place of a report."""
        lines = ["MRI BRAIN - STROKE PROTOCOL", "", "QUALITY CONTROL: FAILED", ""]
        lines.append("No prediction was produced. The input did not pass the checks below,")
        lines.append("and reporting on unreadable data is less safe than declining to report.")
        lines.append("")
        for result in self.critical_failures:
            lines.append(f"  [FAIL] {result.name}: {result.reason}")
        for result in self.warnings:
            lines.append(f"  [WARN] {result.name}: {result.reason}")
        lines.append("")
        lines.append("Please check the input series and resubmit.")
        return "\n".join(lines)


def _as_array(value: Any) -> np.ndarray | None:
    """Return *value* as an ndarray, or None if it is not array-like."""
    if isinstance(value, np.ndarray):
        return value
    return None


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
            Thresholds and tolerances for each QC check. Any key omitted
            falls back to :data:`DEFAULT_CONFIG`.
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    # -- individual gates ---------------------------------------------------

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
        name = "modalities"
        missing = [
            modality
            for modality in self.config["required_modalities"]
            if subject_data.get(modality) is None
        ]
        if missing:
            return QCResult(
                False,
                f"missing required modality: {', '.join(sorted(missing))}",
                name,
            )

        empty = []
        for modality in self.config["required_modalities"]:
            array = _as_array(subject_data[modality])
            if array is None:
                continue
            if array.size == 0:
                empty.append(f"{modality} is empty")
            elif not np.isfinite(array).any():
                empty.append(f"{modality} contains no finite values")
            elif float(np.nanmax(np.abs(array))) == 0.0:
                empty.append(f"{modality} is all zeros")

        if empty:
            return QCResult(False, "; ".join(empty), name)

        return QCResult(True, "all required modalities present", name)

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
        name = "spacing"
        spacing = subject_data.get("spacing")
        if spacing is None:
            metadata = subject_data.get("metadata") or {}
            spacing = metadata.get("spacing")

        if spacing is None:
            return QCResult(False, "voxel spacing is missing from metadata", name)

        try:
            values = [float(s) for s in spacing]
        except (TypeError, ValueError):
            return QCResult(False, f"voxel spacing is not numeric: {spacing!r}", name)

        if len(values) < 3:
            return QCResult(False, f"expected 3 spacing values, got {len(values)}", name)

        values = values[:3]
        lo = self.config["min_spacing_mm"]
        hi = self.config["max_spacing_mm"]

        if not all(np.isfinite(values)):
            return QCResult(False, f"voxel spacing is not finite: {values}", name)
        bad = [v for v in values if v <= 0 or v < lo or v > hi]
        if bad:
            return QCResult(
                False,
                f"voxel spacing {values} outside the plausible range {lo}-{hi} mm",
                name,
            )

        return QCResult(True, f"voxel spacing {values} mm", name)

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
        name = "coverage"
        dwi = _as_array(subject_data.get("dwi"))
        if dwi is None:
            return QCResult(False, "no DWI volume to measure coverage on", name)

        if dwi.ndim != 3:
            return QCResult(False, f"expected a 3-D DWI volume, got {dwi.ndim}-D", name)

        min_shape = tuple(self.config["min_shape"])
        if any(actual < minimum for actual, minimum in zip(dwi.shape, min_shape)):
            return QCResult(
                False,
                f"volume {dwi.shape} is smaller than the minimum {min_shape}",
                name,
            )

        finite = dwi[np.isfinite(dwi)]
        if finite.size == 0:
            return QCResult(False, "DWI contains no finite voxels", name)

        brain_fraction = float((finite > 0).sum() / dwi.size)
        minimum = self.config["min_brain_fraction"]
        if brain_fraction < minimum:
            return QCResult(
                False,
                f"only {brain_fraction:.1%} of the volume is non-zero "
                f"(expected at least {minimum:.0%}); field of view may be empty",
                name,
            )

        return QCResult(True, f"{brain_fraction:.0%} of the volume is non-zero", name)

    def check_dwi_adc_consistency(self, subject_data: dict[str, Any]) -> QCResult:
        """Check that DWI and ADC maps are inversely consistent.

        Lesion regions bright on DWI should be dark on ADC and vice-versa.

        In practice the failure worth catching is the operator error: the same
        series submitted as both DWI and ADC. That makes the ADC-restriction
        finding meaningless while looking perfectly plausible, so it is treated
        as critical.

        Parameters
        ----------
        subject_data:
            Must contain ``dwi`` and ``adc`` arrays plus an optional mask.

        Returns
        -------
        QCResult
        """
        name = "dwi_adc_consistency"
        dwi = _as_array(subject_data.get("dwi"))
        adc = _as_array(subject_data.get("adc"))

        if dwi is None or adc is None:
            return QCResult(True, "skipped: DWI or ADC is not an array", name, WARNING)

        if dwi.shape != adc.shape:
            return QCResult(
                False,
                f"DWI {dwi.shape} and ADC {adc.shape} differ in shape; "
                "they should come from the same acquisition",
                name,
            )

        if np.array_equal(dwi, adc):
            return QCResult(False, "DWI and ADC are identical volumes", name)

        brain = np.isfinite(dwi) & np.isfinite(adc) & ((dwi > 0) | (adc > 0))
        if brain.sum() < 100:
            return QCResult(True, "skipped: too few brain voxels to compare", name, WARNING)

        a = dwi[brain].astype(np.float64)
        b = adc[brain].astype(np.float64)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return QCResult(False, "DWI or ADC has no intensity variation", name)

        correlation = float(np.corrcoef(a, b)[0, 1])
        limit = self.config["max_dwi_adc_correlation"]
        if correlation > limit:
            return QCResult(
                False,
                f"DWI and ADC correlate at r={correlation:.3f} (limit {limit}); "
                "the same series was probably submitted twice",
                name,
            )

        return QCResult(True, f"DWI/ADC correlation r={correlation:.3f}", name)

    # -- orchestration ------------------------------------------------------

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
        results = [self.check_modalities(subject_data)]

        # The remaining gates all read the volumes; if a modality is missing
        # they would fail for a second, less informative reason.
        if results[0].passed:
            results.append(self.check_spacing(subject_data))
            results.append(self.check_coverage(subject_data))
            results.append(self.check_dwi_adc_consistency(subject_data))

        return results

    def evaluate(self, subject_data: dict[str, Any]) -> QCReport:
        """Run every gate and combine the outcome into a :class:`QCReport`."""
        results = self.run_all(subject_data)
        passed = not any(r for r in results if not r.passed and r.severity == CRITICAL)
        return QCReport(passed=passed, results=results)
