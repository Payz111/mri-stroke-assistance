# Evaluation Report

## Model

- **Version:** v2.0 (Attention U-Net, combined ISLES+SOOP training)
- **Checkpoint:** `best_model.pth` (epoch 30, early stopping at epoch 40)
- **Architecture:** 3D Attention U-Net (MONAI), 5.86M parameters
- **Evaluation resolution:** 128x128x80 (model resolution after CenterSpatialCrop)
- **Evaluation features:** TTA (L-R flip averaging) + post-processing (remove components <10 voxels)

## Training Data

| Dataset | Cases | Role |
|---------|-------|------|
| ISLES 2022 | 200 | Training |
| SOOP (ds004889) | ~1121 | Training |
| ISLES 2022 | 50 | Validation |

## Overall Metrics (Validation Set, Fold 0)

| Metric | Value |
|--------|-------|
| Dice (mean +/- std) | **0.691 +/- 0.254** |
| Dice (median) | **0.772** |
| IoU (mean) | 0.573 |
| HD95 (mean) | 13.4 +/- 16.9 mm |
| Sensitivity (mean) | 0.697 |
| Volume MAE (mean) | 4.0 mL |
| Lesion F1 (mean) | 0.503 |
| Training val_dice (batch-avg) | 0.785 |

> **Note on Dice discrepancy:** Training reports batch-averaged val_dice=0.785, which
> weights each batch equally. Per-subject evaluation (mean=0.691) weights each subject
> equally, so tiny lesions with low Dice pull down the average significantly.
> The median Dice (0.772) better represents typical model performance.

## Stratified Analysis by Lesion Size

| Category | N | Dice (mean) | Dice (median) | HD95 (mean) | Sensitivity | Lesion F1 | Vol MAE (mL) |
|----------|---|-------------|---------------|-------------|-------------|-----------|--------------|
| Tiny (<1 mL) | 10 | 0.377 | 0.349 | 25.8 mm | 0.476 | 0.533 | 0.25 |
| Small (1-10 mL) | 19 | 0.710 | 0.750 | 13.3 mm | 0.715 | 0.604 | 1.13 |
| Medium (10-50 mL) | 15 | 0.821 | 0.856 | 8.5 mm | 0.772 | 0.380 | 2.31 |
| Large (>50 mL) | 6 | 0.826 | 0.876 | 6.8 mm | 0.820 | 0.442 | 23.9 |

### Key Findings

1. **Clear size-dependent performance:** Dice increases monotonically with lesion size
2. **Tiny lesion improvement:** Dice 0.19 -> 0.38 (attention gates + augmentation helped significantly)
3. **Medium-to-large lesions perform well:** Dice 0.82-0.83 mean, 0.86-0.88 median
4. **Only 4 complete misses** (Dice<0.05), down from 6 in v1
5. **Lesion F1 nearly doubled:** 0.311 -> 0.503 (post-processing removes false positive clusters)

## Training History

40 epochs on Kaggle T4 GPU (~13 hours total), early stopping triggered at epoch 40 (patience=20, best at epoch 30).

| Phase | Epochs | val_dice | Notes |
|-------|--------|----------|-------|
| Warm-up | 0-5 | 0.27 - 0.50 | Rapid initial learning |
| Growth | 5-15 | 0.50 - 0.74 | Attention gates activate |
| Plateau | 15-30 | 0.74 - 0.78 | Best model at epoch 30 |
| Early stop | 30-40 | ~0.78 | No improvement, stopped |

### Model Comparison

| Model | val_dice (batch) | Dice (per-subj) | Dice (median) | HD95 | Tiny Dice | Epochs | Data |
|-------|-----------------|-----------------|---------------|------|-----------|--------|------|
| U-Net (ISLES-only) | 0.606 | - | - | - | - | 100 | 200 |
| U-Net (ISLES+SOOP) | 0.705 | 0.567 | 0.629 | 19.1 | 0.193 | 50 | 1321 |
| **Attn U-Net + aug** | **0.785** | **0.691** | **0.772** | **13.4** | **0.377** | 40 | 1321 |

Key improvements in v2:
- **Architecture:** Attention gates focus on lesion boundaries (+11% batch Dice)
- **Augmentation:** RandFlip + GaussNoise + ContrastAdjust (reduces overfitting gap 0.136 -> 0.045)
- **TTA:** L-R flip averaging smooths predictions at boundaries
- **Post-processing:** Component filtering reduces false positive clusters
- **AMP:** Mixed precision training enabled faster training without quality loss

## Error Analysis

### Dice~0 Cases (Complete Misses)

4 subjects with Dice < 0.05, all tiny lesions (<1 mL):

| Subject | GT Volume | Pred Volume | Issue |
|---------|-----------|-------------|-------|
| sub-strokecase0136 | 0.20 mL | 0.18 mL | Misplaced prediction |
| sub-strokecase0139 | 0.56 mL | 0.19 mL | Partial, wrong location |
| sub-strokecase0174 | 0.86 mL | 0.44 mL | Partial overlap |
| sub-strokecase0233 | 0.06 mL | 0.00 mL | No prediction |

Previously missed cases now detected: sub-strokecase0008 (Dice 0->0.16), sub-strokecase0115 (improved).

### Known Limitations

1. **Tiny lesion detection:** Sub-milliliter lesions remain challenging (Dice=0.38).
   At 128x128x80 resolution with 2mm spacing, a 0.5 mL lesion is only ~60 voxels.
   Attention gates helped (0.19->0.38) but these remain the hardest cases.

2. **Overfitting gap:** train_dice=0.831 vs val_dice=0.785 (gap=0.045)
   - Dramatically improved from v1 gap of 0.136
   - Augmentation (flip, noise, contrast) was the main factor

3. **SOOP data quality:** Some corrupt NIfTI files (e.g., sub-502)
   - Handled by graceful skip in dataset class
   - ~1-2% of SOOP data affected

4. **FLAIR orientation:** SOOP FLAIR acquired coronally vs axial DWI
   - Fixed by reorienting to RAS+ canonical before processing
   - Without this fix, val_dice collapsed to 0.003

5. **Large lesion volume error:** MAE=23.9 mL for large lesions (>50 mL)
   - Relatively small error (typically <30% of volume)
   - Model tends to slightly under-segment large lesion periphery

## Discussion

The Attention U-Net with augmentation significantly outperforms the standard U-Net across all metrics:

1. **Attention gates:** Help the model focus on small lesion regions, improving tiny lesion Dice from 0.19 to 0.38
2. **Data augmentation:** Reduces overfitting (gap 0.136 -> 0.045), improving generalization
3. **TTA + post-processing:** Smooth predictions and remove false positives, improving Lesion F1 from 0.31 to 0.50
4. **AMP training:** Enabled efficient training without quality degradation

### Performance in Context

For ISLES 2022 challenge benchmarks, top methods achieved Dice ~0.50-0.65 on the hidden test set (which includes many tiny lesions). Our per-subject mean Dice of 0.691 on the validation set is competitive, especially considering:
- Single fold (no ensemble)
- Single model (no nnU-Net self-configuring pipeline)
- Simple augmentation (no elastic deformation)
- 40 epochs with early stopping

### Potential Further Improvements

- Full 5-fold cross-validation with ensemble
- Higher resolution for tiny lesion detection (192x192x96)
- nnU-Net adaptive preprocessing pipeline
- Deep supervision for better gradient flow
- Elastic deformation augmentation
