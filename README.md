# MRI Stroke Assist

AI-powered assistant for ischemic stroke detection and characterization on brain MRI.

Automated pipeline: **DWI + ADC + FLAIR -> lesion segmentation -> structured findings (JSON) -> draft radiology report**.

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| Dice (per-subject mean) | **0.567** | Validation set (50 subjects), fold 0 |
| Dice (median) | **0.629** | Less affected by tiny lesion outliers |
| Dice (batch-averaged) | **0.705** | Training-time metric |
| HD95 | 19.1 mm | 95th percentile Hausdorff distance |
| Model | 3D U-Net (MONAI) | 4.7M parameters |
| Training | 50 epochs | Kaggle T4 GPU, ~10h |
| Training data | ISLES 2022 + SOOP | 1321 cases combined |

**By lesion size:** Tiny 0.19 | Small 0.58 | Medium 0.72 | Large 0.77 (Dice mean)

Training on combined data (ISLES 2022 + SOOP) improved batch-averaged Dice from 0.606 to **0.705** (+16.4%). See [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) for detailed stratified analysis.

## What it does

- **Segments** acute/early-subacute ischemic lesions on DWI/ADC/FLAIR
- **Calculates** volume (mL), max diameter (mm), anatomic location, vascular territory
- **Detects** DWI-FLAIR mismatch (timing marker for <4.5h onset)
- **Checks** ADC restriction (confirms acute ischemia)
- **Generates** structured findings (JSON) and draft radiology reports
- **Validates** reports against findings (5 cross-checks, zero hallucinations)

## What it does NOT do

- Autonomous diagnosis (human-in-the-loop required)
- Treatment recommendations
- Clinical use without proper validation

## Architecture

```
NIfTI files (DWI, ADC, FLAIR)
        |
   Preprocessing
   - Resample FLAIR to DWI space
   - Z-score normalization
   - Pad/crop to 128x128x80
        |
   3D U-Net (MONAI)
   - 4 levels: 32->64->128->256
   - DiceFocalLoss training
        |
   Binary lesion mask
        |
   Findings extraction
   - Volume & diameter
   - Laterality (L/R/bilateral)
   - Anatomic location (heuristic)
   - Vascular territory (MCA/ACA/PCA/VB)
   - ADC restriction check
   - DWI-FLAIR mismatch detection
        |
   Structured JSON (V1Findings)
        |
   Template-based report
   - Zero hallucinations
   - 5 validation checks
```

## Project Structure

```
src/
  data/          Dataset class + MONAI transforms
  preprocess/    Intensity normalization, registration
  models/        3D U-Net architecture + losses (Dice, Focal, DiceFocal)
  train/         Trainer, callbacks (checkpoint, early stopping)
  eval/          Metrics (Dice, IoU, HD95, lesion F1), stratified evaluation
  findings/      Structured findings extraction (volume, location, territory, mismatch)
  report/        Template-based report generation + validator
  inference/     End-to-end inference pipeline + visualization
configs/         YAML configurations (model, training, augmentation)
scripts/         CLI entry points (train, evaluate, smoke tests)
demo/            Gradio web demo
notebooks/       EDA, Kaggle training, evaluation notebooks
data/splits/     5-fold cross-validation splits
```

## Quick Start

### Prerequisites

- Python 3.11+ (3.11 recommended for Gradio demo)
- No GPU required for inference (CPU supported)

### Installation

```bash
git clone git@gitlab.com:Payz111/mri-stroke-assistance.git
cd mri-stroke-assistance
pip install -e ".[dev]"
```

### Run Demo (no data needed)

```bash
python demo/app.py
# Open http://localhost:7860
# Click "Synthetic Demo" for instant results
```

### Train from scratch

```bash
# 1. Download ISLES 2022 dataset from Zenodo
#    https://zenodo.org/record/7153326
#    Extract to data/raw/isles22/ISLES-2022/

# 2. Train (GPU recommended, see notebooks/ for Kaggle notebooks)
python scripts/train.py --fold 0 --epochs 50 --device cuda

# 3. Evaluate
python scripts/evaluate.py --checkpoint outputs/fold_0/checkpoints/best_model.pth --fold 0

# 4. Test findings + report pipeline
python scripts/smoke_test_findings.py --synthetic
```

For combined training with SOOP dataset, see `notebooks/03_kaggle_combined_training.ipynb`.

## Example Report Output

```
MRI BRAIN - STROKE PROTOCOL
Study ID: sub-strokecase0042
Model: v1.0-unet3d-isles22

TECHNIQUE:
MRI brain with stroke protocol. Sequences available: DWI, ADC, FLAIR.

FINDINGS:
1 ischemic lesion(s) identified, total volume 8.5 mL.

Lesion 1: 8.5 mL (25 mm max diameter), left hemispheric temporal,
MCA territory. ADC restricted (confirms acute ischemia).
FLAIR signal subtle/equivocal.
DWI-FLAIR mismatch: positive (suggests <4.5h onset).

IMPRESSION:
Acute ischemic infarct, left, total volume 8.5 mL (confidence: 80%).
```

## Datasets

| Dataset | Source | Modalities | Cases | Role |
|---------|--------|-----------|-------|------|
| [ISLES 2022](https://zenodo.org/record/7153326) | Zenodo | DWI, ADC, FLAIR | 250 | Train (200) + Val (50), 5-fold CV |
| [SOOP](https://openneuro.org/datasets/ds004889) | OpenNeuro | DWI, ADC, FLAIR | 1121 | Training only |

ISLES 2022 key statistics:
- Median lesion volume: 6.66 mL
- Size distribution: 43 tiny (<1mL), 95 small (1-10mL), 79 medium (10-50mL), 30 large (>50mL)
- 3 cases with empty masks (negative controls)

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Zero hallucinations** | Report text generated ONLY from structured JSON findings |
| **Human-in-the-loop** | Physician always makes the final decision |
| **Evidence-linked** | Every claim tied to a mask and slice indices |
| **Fail-safe** | QC gate: poor quality input -> no prediction |
| **Reproducible** | Fixed seeds, 5-fold CV, YAML configs |

## Tech Stack

- **Deep Learning:** PyTorch, MONAI
- **Medical Imaging:** nibabel, SimpleITK, scipy
- **Training:** AdamW, CosineAnnealingLR, DiceFocalLoss
- **Evaluation:** Dice, IoU, HD95, lesion-wise F1, stratified by size
- **Demo:** Gradio
- **Config:** YAML (Hydra-ready)

## License

MIT

## Author

Rinat -- Neurologist & ML Engineer
