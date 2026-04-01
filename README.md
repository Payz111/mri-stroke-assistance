# MRI Stroke Assist

AI-powered assistant for ischemic stroke detection and characterization on brain MRI.

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/Paizutdin/mri-stroke-assist)** — try the full pipeline in your browser (no setup needed).

Automated pipeline: **DWI + ADC + FLAIR -> lesion segmentation -> structured findings (JSON) -> draft radiology report**.

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| Dice (per-subject mean) | **0.691** | Validation set (50 subjects), fold 0 |
| Dice (median) | **0.772** | Less affected by tiny lesion outliers |
| HD95 | 13.4 mm | 95th percentile Hausdorff distance |
| Sensitivity | 0.697 | True positive rate |
| Lesion F1 | 0.503 | Lesion-wise detection score |
| Model | 3D Attention U-Net (MONAI) | 5.86M parameters |
| Training | 40 epochs (AMP) | Kaggle T4 GPU |
| Training data | ISLES 2022 + SOOP | 1321 cases combined |

**By lesion size:** Tiny 0.38 | Small 0.71 | Medium 0.82 | Large 0.83 (Dice mean)

See [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) for detailed stratified analysis.

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
   3D Attention U-Net (MONAI)
   - 4 levels: 32->64->128->256
   - Attention gates at skip connections
   - DiceFocalLoss + augmentation + AMP
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

## Deployment

### Docker

```bash
# API server (port 8000)
docker compose up api

# Gradio demo (port 7860)
docker compose up demo
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Model info |
| GET | `/health` | Health check |
| POST | `/predict` | Upload ZIP with DWI/ADC/FLAIR -> get JSON findings + report |

### HuggingFace Spaces

Live at: https://huggingface.co/spaces/Paizutdin/mri-stroke-assist

Deploy your own:
```bash
bash scripts/deploy_hf.sh <your-hf-username> [checkpoint-path]
```

## Tech Stack

- **Deep Learning:** PyTorch, MONAI
- **Medical Imaging:** nibabel, SimpleITK, scipy
- **Training:** AdamW, CosineAnnealingLR, DiceFocalLoss
- **Evaluation:** Dice, IoU, HD95, lesion-wise F1, stratified by size
- **Serving:** FastAPI (REST API), Gradio (web demo), Docker
- **Deployment:** HuggingFace Spaces, Docker Compose
- **Config:** YAML (Hydra-ready)

## License

MIT

## Author

Rinat -- Neurologist & ML Engineer
