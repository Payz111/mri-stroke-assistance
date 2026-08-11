"""Pydantic models for structured findings JSON output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Laterality(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"


class VascularTerritory(str, Enum):
    MCA = "MCA"
    ACA = "ACA"
    PCA = "PCA"
    VERTEBROBASILAR = "vertebrobasilar"
    WATERSHED = "watershed"
    UNCERTAIN = "uncertain"


class AnatomicLocation(str, Enum):
    FRONTAL = "frontal"
    PARIETAL = "parietal"
    TEMPORAL = "temporal"
    OCCIPITAL = "occipital"
    INSULA = "insula"
    BASAL_GANGLIA = "basal_ganglia"
    THALAMUS = "thalamus"
    INTERNAL_CAPSULE = "internal_capsule"
    BRAINSTEM = "brainstem"
    CEREBELLUM = "cerebellum"
    MULTIPLE = "multiple"


class FlairSignal(str, Enum):
    NONE = "none"
    SUBTLE = "subtle"
    DEFINITE = "definite"


class MismatchStatus(str, Enum):
    YES = "yes"
    NO = "no"
    INDETERMINATE = "indeterminate"


class ImpressionClass(str, Enum):
    NO_ACUTE_INFARCT = "no_acute_infarct"
    ACUTE = "acute"
    EARLY_SUBACUTE = "early_subacute"
    INDETERMINATE = "indeterminate"


# ---------- Sub-models ----------


class QualityGate(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class EvidenceSlices(BaseModel):
    series_uid: str | None = None
    slice_indices: list[int] = Field(default_factory=list)


class ProtocolPresent(BaseModel):
    dwi: bool
    adc: bool
    flair: bool
    gre_swi: bool = False
    tof_mra: bool = False


class Lesion(BaseModel):
    lesion_id: int
    mask_ref: str
    volume_ml: float = Field(ge=0)
    max_diameter_mm: float = Field(ge=0)
    laterality: Laterality
    anatomic_location: list[AnatomicLocation]
    vascular_territory: VascularTerritory
    territory_confidence: float = Field(ge=0, le=1)
    adc_low_confirmed: bool
    adc_confidence: float = Field(ge=0, le=1)
    flair_hyperintensity: FlairSignal
    flair_confidence: float = Field(ge=0, le=1)
    dwi_flair_mismatch: MismatchStatus
    mismatch_confidence: float = Field(ge=0, le=1)
    evidence_slices: EvidenceSlices


# ---------- V1 top-level ----------


class V1Findings(BaseModel):
    """Complete V1 findings output."""

    study_id: str
    model_version: str
    timestamp: str  # ISO 8601

    protocol_present: ProtocolPresent
    quality_gate: QualityGate

    lesions: list[Lesion] = Field(default_factory=list)
    total_lesion_count: int = Field(ge=0)
    total_lesion_volume_ml: float = Field(ge=0)

    overall_impression: ImpressionClass
    overall_confidence: float = Field(ge=0, le=1)

    combined_mask_ref: str


# ---------- V2 Perfusion additions ----------


class TargetMismatchStatus(str, Enum):
    TARGET_MISMATCH = "target_mismatch"
    NO_MISMATCH = "no_mismatch"
    INDETERMINATE = "indeterminate"


class PerfusionMapsPresent(BaseModel):
    tmax: bool
    cbf: bool
    cbv: bool = False
    mtt: bool = False


class PerfusionThresholds(BaseModel):
    hypoperfusion_tmax_sec: float = 6.0
    core_rcbf_threshold: float = 0.30


class CoreFindings(BaseModel):
    volume_ml: float = Field(ge=0)
    mask_ref: str
    method: str = "rCBF_threshold"
    confidence: float = Field(ge=0, le=1)


class HypoperfusionFindings(BaseModel):
    volume_ml: float = Field(ge=0)
    mask_ref: str
    method: str = "Tmax_threshold"
    confidence: float = Field(ge=0, le=1)


class MismatchMetrics(BaseModel):
    mismatch_volume_ml: float = Field(ge=0)
    mismatch_ratio: float = Field(ge=0)
    target_mismatch_status: TargetMismatchStatus
    criteria_used: dict
    confidence: float = Field(ge=0, le=1)


class V2PerfusionFindings(BaseModel):
    """V2 perfusion findings (extends V1)."""

    perfusion_source: str  # "CTP" or "MR_PWI"
    maps_present: PerfusionMapsPresent
    thresholds_used: PerfusionThresholds
    perfusion_quality_gate: QualityGate

    core: CoreFindings
    hypoperfusion: HypoperfusionFindings
    mismatch: MismatchMetrics
