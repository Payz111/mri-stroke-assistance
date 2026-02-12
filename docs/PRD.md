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
- Dice score > 0.60 (mean) on ISLES 2022
- Lesion-wise recall > 0.85
- Zero hallucinations in generated reports
- QC gates block poor-quality inputs

## Detailed requirements

See the full development plan for comprehensive specifications.
