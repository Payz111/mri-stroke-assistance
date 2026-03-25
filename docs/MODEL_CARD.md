# Model Card: MRI Stroke Segmentation v1.0

## Model Details

- **Architecture:** 3D U-Net (MONAI)
- **Parameters:** 4.7M (feature channels: 32, 64, 128, 256)
- **Checkpoint size:** 19 MB
- **Input:** DWI + ADC + FLAIR (3-channel, 3D volume)
- **Output:** Lesion probability map (sigmoid, threshold 0.5)
- **Dropout:** 0.1

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
| Loss | DiceFocalLoss (dice_weight=0.5, focal_weight=0.5) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Batch size | 4 |
| Hardware | Kaggle T4 GPU (16 GB) |
| Training time | ~10 hours |
| Spatial padding | DivisiblePad(k=16) |
| Normalization | Z-score (per-modality, non-zero voxels) |

### Preprocessing

- Resample FLAIR to DWI space (scipy.ndimage.zoom)
- Reorient to RAS+ canonical orientation (nibabel)
- Z-score intensity normalization per modality
- Pad to divisible-by-16 spatial dimensions

### Augmentations (training only)

- Random affine (rotation, scaling)
- Random intensity shift

## Metrics

| Metric | Value |
|--------|-------|
| **Dice (validation)** | **0.705** |
| Dice (training) | 0.841 |
| Validation loss | 0.256 |
| Training loss | 0.193 |

Best model saved at epoch 35 (by validation Dice).

### Training Dynamics

- Rapid learning in epochs 0-20 (Dice: 0.01 -> 0.70)
- Plateau reached at epoch ~35
- Train-val Dice gap: 0.14 (moderate overfitting)
- Train-val loss gap: 0.07

## Intended Use

- Research and development
- Clinical decision support (with physician oversight)
- Educational purposes

## Limitations

- Validated on ISLES 2022 only (multi-center European data)
- May underperform on:
  - Very small lesions (< 1 mL)
  - Posterior fossa lesions
  - Non-standard MRI protocols or scanners not represented in training
- Moderate overfitting gap (0.14) suggests room for regularization improvement
- Single fold evaluation (fold 0); full 5-fold CV pending
- NOT validated for clinical use
- NOT a medical device

## Ethical Considerations

- Requires human expert review for all outputs
- Should not be used for autonomous diagnosis
- No patient data retained in model weights
- Reports include explicit disclaimer: "FOR RESEARCH USE ONLY"
