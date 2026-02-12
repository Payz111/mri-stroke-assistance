# MRI Stroke Assist

AI-powered assistant for ischemic stroke detection and characterization on brain MRI.

## What it does

- **Segments** acute/early-subacute ischemic lesions on DWI/ADC/FLAIR
- **Calculates** volume, anatomic location, vascular territory
- **Detects** DWI-FLAIR mismatch (timing marker)
- **Generates** structured findings (JSON) and draft text reports with **zero hallucinations**
- **V2:** Perfusion-based core/penumbra/target mismatch analysis (CTP)

## What it does NOT do

- Autonomous diagnosis (human-in-the-loop required)
- Treatment recommendations
- Clinical use without proper validation

## Project Structure

```
src/
  data/          # Dataset classes and transforms
  preprocess/    # MRI preprocessing pipeline
  qc/            # Quality control gates
  models/        # Segmentation architectures (3D U-Net)
  train/         # Training loop
  postprocess/   # Post-processing and FP filtering
  findings/      # Structured findings extraction
  report/        # Report generation and validation
  eval/          # Metrics and error analysis
  inference/     # Inference pipeline and API
  v2_perfusion/  # CT Perfusion module
configs/         # Hydra configuration files
scripts/         # Entry points (train, evaluate, infer)
demo/            # Gradio web demo
tests/           # Unit and integration tests
docs/            # Documentation
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Download data
bash scripts/download_isles22.sh

# Preprocess
python scripts/preprocess_dataset.py --config configs/default.yaml

# Train
python scripts/train.py --config-name baseline

# Evaluate
python scripts/evaluate.py --config-name default

# Demo
python demo/app.py
```

## Datasets

| Dataset | Version | Modalities | Cases |
|---------|---------|------------|-------|
| [ISLES 2022](https://zenodo.org/record/7153326) | V1 | DWI, ADC, FLAIR | 250 |
| [ISLES 2024](https://isles-24.grand-challenge.org/) | V2 | CTP (Tmax/CBF/CBV/MTT) | 149 |

## Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| Human-in-the-loop | Physician always makes the final decision |
| Zero hallucinations | Text generated ONLY from structured JSON findings |
| Evidence-linked | Every claim tied to a mask/slice |
| Fail-safe | Poor quality input -> no prediction, QC report only |

## Tech Stack

PyTorch, MONAI, nibabel, SimpleITK, Hydra, MLflow, Gradio, FastAPI, Docker

## License

MIT

## Author

Rinat — Neurologist & ML Engineer
