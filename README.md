# MRI Stroke Assist

[![CI](https://github.com/Payz111/mri-stroke-assistance/actions/workflows/ci.yml/badge.svg)](https://github.com/Payz111/mri-stroke-assistance/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AI-powered assistant for ischemic stroke detection and characterization on brain MRI.

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/Paizutdin/mri-stroke-assist)** —
runs the released Attention U-Net on CPU. Upload your own DWI/ADC/FLAIR NIfTI files to get a
mask, structured findings and a draft report, or press *Synthetic Demo* to see the
findings → report → validation chain with no data of your own.

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

> **How to read these numbers.** They come from the **validation split of fold 0**
> (50 ISLES 2022 subjects), with TTA and small-component filtering applied — not from a
> held-out test set, and not averaged over all 5 folds. Only fold 0 has been trained so
> far, so there is no cross-fold variance estimate yet. They are therefore **not directly
> comparable** to ISLES 2022 challenge leaderboard scores, which are computed on a hidden
> test set. Reproduce with:
> `python scripts/evaluate.py --checkpoint <ckpt> --fold 0`

See [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) for the detailed stratified analysis, and
[Training_results/](Training_results/) for the raw artifacts behind every number — training
curves, per-subject metrics and prediction overlays for each run, including the failed ones.

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
   QC gate
   - Modalities present and non-empty
   - Plausible voxel spacing and field of view
   - DWI/ADC not the same series
   - Critical failure -> no prediction at all
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
  data/            Dataset classes (ISLES 2022, SOOP, combined) + MONAI transforms
  preprocess/      Intensity normalization, resampling, registration
  models/          3D U-Net and Attention U-Net + losses (Dice, Focal, DiceFocal)
  train/           Trainer, callbacks (checkpoint, early stopping)
  eval/            Metrics (Dice, IoU, HD95, lesion F1), stratified evaluation
  findings/        Structured findings extraction (volume, location, territory, mismatch)
  report/          Template-based report generation + validator
  inference/       End-to-end pipeline, TTA, REST API, visualization
  v2_perfusion/    CT perfusion core/penumbra/mismatch (prototype)
  qc/              Quality-control gate (specified, not implemented)
configs/           YAML configurations (model, training, augmentation)
scripts/           CLI entry points (train, evaluate, infer, smoke tests)
demo/              Gradio web demo
notebooks/         EDA, Kaggle training and evaluation notebooks
docs/              Architecture, decision records, model card, evaluation report
Training_results/  Curves, metrics and overlays for every run
data/splits/       5-fold cross-validation splits
```

## Documentation

| Document | What it covers |
|----------|----------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline stages, module responsibilities, what is and is not implemented |
| [DECISIONS.md](docs/DECISIONS.md) | 12 architecture decision records — the alternatives, the reasoning, and how each decision turned out |
| [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) | Stratified metrics, error analysis, known limitations |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use, training data, performance, ethical considerations |
| [PRD.md](docs/PRD.md) | Original product requirements |

## Quick Start

### Prerequisites

- Python 3.11+ (3.11 recommended for Gradio demo)
- No GPU required for inference (CPU supported)

### Installation

```bash
git clone https://github.com/Payz111/mri-stroke-assistance.git
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

# 3. Evaluate (TTA + small-component filtering are on by default)
python scripts/evaluate.py \
    --checkpoint outputs/fold_0/checkpoints/best_model.pth \
    --fold 0

# 4. Test findings + report pipeline
python scripts/smoke_test_findings.py --synthetic
```

### Run on a single case

```bash
python scripts/infer_single.py \
    --input data/raw/isles22/ISLES-2022/sub-strokecase0001 \
    --output outputs/case0001 \
    --checkpoint outputs/fold_0/checkpoints/best_model.pth
```

Writes `prediction_mask.nii.gz`, `findings.json` and `report.txt`.

> **Checkpoint / architecture must match.** `configs/default.yaml` describes the released
> Attention U-Net, so the commands above need no `--config`. To load the plain `unet3d`
> baseline instead, pass `--config configs/experiment/baseline.yaml` — a checkpoint loaded
> against the wrong architecture fails in `load_state_dict`.

For combined training with SOOP dataset, see `notebooks/03_kaggle_combined_training.ipynb`.

## Example Report Output

```
MRI BRAIN - STROKE PROTOCOL
Study ID: sub-strokecase0042
Model: v2.0-attention-unet3d-isles22-soop

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

| Principle | Status |
|-----------|--------|
| **Zero hallucinations** | Implemented — report text generated ONLY from structured JSON findings, then cross-checked by [validator.py](src/report/validator.py) |
| **Human-in-the-loop** | By design — the tool outputs a *draft*; the physician decides |
| **Evidence-linked** | Implemented — every lesion carries a mask ref and slice indices |
| **Reproducible** | Partial — fixed seeds, YAML configs, 5-fold splits generated; only fold 0 trained so far |
| **Fail-safe** | Implemented — [qc_pipeline.py](src/qc/qc_pipeline.py) gates every study before the model; a critical failure returns no mask and no findings, only an explanation ([ADR-006](docs/DECISIONS.md)) |

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
- **Training:** AdamW, CosineAnnealingLR, DiceFocalLoss, AMP
- **Evaluation:** Dice, IoU, HD95, lesion-wise F1, stratified by size
- **Serving:** FastAPI (REST API), Gradio (web demo), Docker
- **Deployment:** HuggingFace Spaces, Docker Compose
- **Config:** plain YAML + argparse

> Experiment tracking (MLflow) and Hydra config composition were planned but are
> **not wired up** — training reads YAML directly and writes `training_history.json`.

## License

MIT

## Author

Paizutdin Mugutdinov -- Neurologist & ML Engineer
