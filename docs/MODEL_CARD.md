# Model Card: MRI Stroke Segmentation v2.0

## Model Details

- **Architecture:** 3D Attention U-Net (MONAI)
- **Parameters:** 5.86M (feature channels: 32, 64, 128, 256)
- **Checkpoint size:** ~23 MB
- **Input:** DWI + ADC + FLAIR (3-channel, 3D volume)
- **Output:** Lesion probability map (sigmoid, threshold 0.5)
- **Dropout:** 0.1
- **Attention:** Gates at skip connections (focus on lesion boundaries)

## Training Data

| Dataset | Source | Cases | Role |
|---------|--------|-------|------|
| ISLES 2022 | Zenodo | 200 (fold 0) | Train |
| SOOP (ds004889) | OpenNeuro | ~1121 | Train |
| ISLES 2022 | Zenodo | 50 (fold 0) | Validation |

**Total training cases:** ~1321
**Validation cases:** 50 (ISLES 2022 only, stratified by lesion size)

## Training Details

| Parameter | Value |
|-----------|-------|
| Loss | DiceFocalLoss (dice_weight=0.4, focal_weight=0.6) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR |
| Epochs | 40 (early stopping, patience=20) |
| Batch size | 4 |
| Mixed precision | AMP (torch.amp.autocast + GradScaler) |
| Hardware | Kaggle T4 GPU (16 GB) |
| Training time | ~13 hours |
| Normalization | Z-score (per-modality, non-zero voxels) |

### Preprocessing

- Resample FLAIR to DWI space (scipy.ndimage.zoom)
- Reorient to RAS+ canonical orientation (nibabel)
- Z-score intensity normalization per modality
- Pad/crop to 128x128x80

### Augmentations (training only)

- RandFlip L-R (prob=0.5) and A-P (prob=0.3)
- RandGaussianNoise (prob=0.2, std=0.05)
- RandAdjustContrast (prob=0.2, gamma=0.7-1.5)

### Inference Enhancements

- Test-time augmentation: L-R flip averaging (2 forward passes)
- Post-processing: Remove connected components < 10 voxels

## Metrics

### Per-Subject Evaluation (50 validation subjects)

| Metric | Value |
|--------|-------|
| **Dice (mean)** | **0.691** |
| **Dice (median)** | **0.772** |
| IoU (mean) | 0.573 |
| HD95 (mean) | 13.4 mm |
| Sensitivity | 0.697 |
| Lesion F1 | 0.503 |

### By Lesion Size (Dice mean)

| Tiny (<1 mL) | Small (1-10 mL) | Medium (10-50 mL) | Large (>50 mL) |
|---------------|-----------------|-------------------|----------------|
| 0.38 | 0.71 | 0.82 | 0.83 |

### Training Dynamics

- Best model at epoch 30 (by validation Dice)
- Train-val Dice gap: 0.045 (low overfitting)
- Batch-averaged val_dice: 0.785

## Intended Use

- Research and development
- Clinical decision support (with physician oversight)
- Educational purposes

## Limitations

- Validated on ISLES 2022 only (multi-center European data)
- May underperform on:
  - Very small lesions (< 1 mL): Dice=0.38
  - Posterior fossa lesions
  - Non-standard MRI protocols or scanners not represented in training
- Single fold evaluation (fold 0); full 5-fold CV pending
- NOT validated for clinical use
- NOT a medical device

## Ethical Considerations

- Requires human expert review for all outputs
- Should not be used for autonomous diagnosis
- No patient data retained in model weights
- Reports include explicit disclaimer: "FOR RESEARCH USE ONLY"
