# Architecture — MRI Stroke Assist

> This document describes the **target** architecture. Sections marked
> *[NOT IMPLEMENTED]* are specified but currently exist only as stubs — see
> the implementation status notes below before relying on them.

## Pipeline Overview

```
Input (NIfTI) -> Preprocessing -> [QC Gates] -> Segmentation Model
    -> Postprocessing -> Structured Findings -> Report Generation -> Output
```

The shipped path (`src/inference/pipeline.py`, used by the demo and the REST API)
currently runs: Preprocessing -> Model -> Findings -> Report -> Validation.
QC gating is skipped entirely.

## Modules

### Preprocessing
- DICOM to NIfTI conversion (if needed)
- Orientation normalization (RAS)
- Resampling to common spacing
- Co-registration (DWI <-> ADC <-> FLAIR)
- Brain extraction (optional)
- Intensity normalization per modality

### QC Gates *[NOT IMPLEMENTED]*
`src/qc/qc_pipeline.py` raises `NotImplementedError`; nothing calls it. Planned checks:
- Missing modality check
- Spacing/size sanity
- Coverage check (posterior fossa)
- DWI-ADC consistency check
- FAIL = no-score, return QC report only

### Segmentation Model
- 3D U-Net (baseline)
- Input: (DWI, ADC, FLAIR) tensor, shape (3, D, H, W)
- Output: lesion probability map

### Postprocessing
Implemented: thresholding + binarization (0.5), connected-component split in
`src/findings/builder.py`, and min-size filtering in `src/eval/postprocess.py`
(`remove_small_components`, default 10 voxels, applied by `scripts/evaluate.py`).

*[NOT IMPLEMENTED]* — `src/postprocess/` (`thresholding.py`, `connected_components.py`,
`fp_filter.py`, `lesion_extraction.py`) is a stub package. DWI-ADC consistency filtering
and edge exclusion do not exist yet.

### Structured Findings
- Volume, max diameter
- Laterality, anatomic location, vascular territory
- FLAIR hyperintensity, DWI-FLAIR mismatch
- Evidence slices

### Report Generation
- Template-based text generation from JSON
- Validator ensures zero hallucinations

## V2: Perfusion Module
- CTP maps (Tmax, CBF, CBV, MTT)
- Threshold-based core/penumbra segmentation
- Target mismatch calculation

See `src/findings/schema.py` for the complete data contract.
