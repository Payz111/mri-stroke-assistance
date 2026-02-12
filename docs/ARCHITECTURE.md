# Architecture — MRI Stroke Assist

## Pipeline Overview

```
Input (NIfTI) -> Preprocessing -> QC Gates -> Segmentation Model
    -> Postprocessing -> Structured Findings -> Report Generation -> Output
```

## Modules

### Preprocessing
- DICOM to NIfTI conversion (if needed)
- Orientation normalization (RAS)
- Resampling to common spacing
- Co-registration (DWI <-> ADC <-> FLAIR)
- Brain extraction (optional)
- Intensity normalization per modality

### QC Gates
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
- Thresholding + binarization
- Connected component analysis
- False positive filtering (DWI-ADC consistency, min size, edge exclusion)

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
