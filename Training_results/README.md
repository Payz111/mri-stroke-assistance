# Training Results

Raw artifacts from every training and evaluation run, kept as evidence behind the
numbers in [docs/EVALUATION_REPORT.md](../docs/EVALUATION_REPORT.md).

**Model weights (`*.pth`) are deliberately not tracked in git** — they are 18–23 MB
each and `.gitignore` excludes them. What is committed is the lightweight evidence:
training curves, per-epoch histories, per-subject metrics and qualitative overlays
(~4.5 MB total).

Directories are named `Train_<MM_DD_YYYY>` after the date of the run.

## Experiment history

| Directory | Experiment | Train data | Epochs | best val_dice (batch-avg) |
|-----------|-----------|-----------|--------|---------------------------|
| `Train_02_26_2026` | 3D U-Net baseline, ISLES 2022 only | 200 | 100 | 0.606 |
| `Train_03_01_2026` | U-Net + augmentation v1 — **reverted**, augmentation far too aggressive | 200 | 56 | 0.405 |
| `Train_03_10_2026` | U-Net, ISLES 2022 + SOOP combined | 1321 | 50 | 0.705 |
| `Train_03_27_2026` | **Current best** — Attention U-Net + mild augmentation + AMP | 1321 | 40 (best @ 30) | **0.785** |

The `Train_03_01_2026` run is kept on purpose. Augmentation that looked reasonable on
paper cost 20 points of Dice, and the failure is more instructive than the successes.

## Evaluation runs

Per-subject evaluation (each subject weighted equally), 50 ISLES 2022 validation
subjects from fold 0.

| Directory | Checkpoint evaluated | Dice (mean) | Dice (median) | HD95 |
|-----------|---------------------|-------------|---------------|------|
| `Train_03_23_2026` | `mri-stroke-model-v1` | 0.303 | 0.304 | 24.4 mm |
| `Train_03_23_2026_v2` | `mri-stroke-model` (combined U-Net) | 0.567 | 0.629 | 19.1 mm |
| `eval_train_03_27_2026` | `train-res-03-27-2026` (Attention U-Net) | **0.691** | **0.772** | **13.4 mm** |

`Train_03_23_2026` evaluated a different, earlier checkpoint than `_v2` and is
superseded by it; it is kept for completeness only. The `_v2` numbers are the ones
quoted as "U-Net (ISLES+SOOP)" in the evaluation report, and `eval_train_03_27_2026`
is the current headline result.

## Contents of a run directory

| File | What it is |
|------|-----------|
| `training_history.json` | per-epoch train/val loss and Dice |
| `training_curves.png` | those histories plotted |
| `experiment_meta.json` | architecture, loss, augmentation, LR, batch size (03_27 only) |
| `eval_results.json` | overall + per-subject metrics |
| `eval_metrics.png` | metric distributions across the validation set |
| `overlay_sub-*.png` | prediction vs ground truth on individual subjects |
| `checkpoints/best_model.pth` | weights — **not in git** |

## Reproducing

```bash
python scripts/evaluate.py \
    --checkpoint Training_results/Train_03_27_2026/checkpoints/best_model.pth \
    --config configs/experiment/attention_aug.yaml \
    --fold 0
```

Training was run on Kaggle T4 GPUs; see the notebooks in [../notebooks/](../notebooks/).
