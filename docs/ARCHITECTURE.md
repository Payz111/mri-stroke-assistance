# Architecture — MRI Stroke Assist

> This document describes the **target** architecture. Sections marked
> *[NOT IMPLEMENTED]* are specified but currently exist only as stubs — see
> the implementation status notes below before relying on them.

## Pipeline Overview

```
Input (NIfTI) -> QC Gates -> Preprocessing -> Segmentation Model
    -> Postprocessing -> Structured Findings -> Report Generation -> Output
                              |
                     QC critical failure
                              |
                    no prediction, QC report only
```

The shipped path (`src/inference/pipeline.py`, used by the demo and the REST API)
runs the QC gate first: on a critical failure it returns no mask and no findings,
only an explanation of which check failed.

## Modules

### Preprocessing
- DICOM to NIfTI conversion (if needed)
- Orientation normalization (RAS)
- Resampling to common spacing
- Co-registration (DWI <-> ADC <-> FLAIR)
- Brain extraction (optional)
- Intensity normalization per modality

### QC Gates
`src/qc/qc_pipeline.py`, enforced in `run_inference` before the model runs:
- `modalities` — DWI/ADC/FLAIR present, non-empty, not all-zero, not all-NaN
- `spacing` — present, numeric, finite, within 0.3–6.0 mm
- `coverage` — 3-D, at least 32×32×8 voxels, at least 2% non-zero
- `dwi_adc_consistency` — DWI and ADC not the same series (r ≤ 0.98), shapes match
- Critical failure = no-score: no mask, no findings, QC report only

Measured on all 250 ISLES 2022 subjects: 0 blocked, 0.0% false positives.
Pass `run_qc=False` to `run_inference` to bypass the gate when inspecting raw
model behaviour.

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
