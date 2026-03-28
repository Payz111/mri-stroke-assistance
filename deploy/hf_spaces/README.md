---
title: MRI Stroke Assist
emoji: "\U0001F9E0"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered ischemic stroke detection on brain MRI
---

# MRI Stroke Assist

AI-powered assistant for ischemic stroke detection and characterization on brain MRI.

**Pipeline:** DWI + ADC + FLAIR -> lesion segmentation -> structured findings (JSON) -> draft radiology report.

## How to use

1. **Synthetic Demo** (no data needed): Click "Synthetic Demo" to see the full pipeline in action with generated data.
2. **Real inference**: Upload a ZIP archive containing DWI, ADC, and FLAIR NIfTI files (.nii or .nii.gz).

## Model

- **Architecture:** 3D Attention U-Net (MONAI), 5.86M parameters
- **Training data:** ISLES 2022 (250 cases) + SOOP (1121 cases)
- **Dice score:** 0.691 (per-subject mean), 0.772 (median)
- **Inference:** CPU-only, ~30-60 seconds per case

## Disclaimer

**FOR RESEARCH USE ONLY.** This is a portfolio project, not a medical device.
All results require expert review. Not validated for clinical use.

## Links

- [Source code (GitLab)](https://gitlab.com/Payz111/mri-stroke-assistance)
- [Evaluation Report](https://gitlab.com/Payz111/mri-stroke-assistance/-/blob/master/docs/EVALUATION_REPORT.md)
