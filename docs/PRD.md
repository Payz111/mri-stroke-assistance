# Product Requirements Document — MRI Stroke Assist

## Overview

AI-powered clinical decision support system for ischemic stroke detection and characterization on brain MRI.

## Problem Statement

Acute ischemic stroke requires rapid and accurate MRI interpretation. Radiologists and neurologists need assistance with:
- Lesion detection and segmentation
- Volume quantification
- Anatomic and vascular territory mapping
- Timing assessment (DWI-FLAIR mismatch)
- Structured reporting

## Scope

### V1: Core Pipeline (DWI/ADC/FLAIR)
- Input: DWI, ADC, FLAIR NIfTI volumes
- Output: Lesion mask, structured findings JSON, draft text report

### V2: Perfusion Module (CTP)
- Input: CT Perfusion maps (Tmax, CBF, CBV, MTT)
- Output: Core/penumbra volumes, mismatch metrics

## Out of Scope
- Treatment recommendations
- Autonomous diagnosis
- Real-time PACS integration
- Regulatory approval

## Success Criteria

Status as of the current model (Attention U-Net, fold 0 validation — see
[EVALUATION_REPORT.md](EVALUATION_REPORT.md)):

| Criterion | Target | Actual | Met |
|-----------|--------|--------|-----|
| Dice (per-subject mean) on ISLES 2022 | > 0.60 | 0.691 | Yes |
| Lesion-wise detection | recall > 0.85 | lesion F1 0.503, voxel sensitivity 0.697 | **No** |
| Zero hallucinations in generated reports | required | template-only text, 5 validator cross-checks | Yes |
| QC gates block poor-quality inputs | required | 4 gates enforced before the model; 0/250 false positives | Yes |

The one unmet criterion is the honest state of the project, not an oversight.
Lesion-wise detection is dominated by sub-millilitre lesions, where Dice is 0.377 —
see the limitations section of [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

## Roadmap

1. Complete 5-fold cross-validation and report cross-fold variance
2. Improve tiny-lesion detection (higher input resolution, deep supervision)
3. Validate the V2 CT perfusion pipeline against real ISLES 2024 data
