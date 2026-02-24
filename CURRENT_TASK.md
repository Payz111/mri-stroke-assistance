# Текущая задача

> Этот файл содержит детали задачи, над которой сейчас работаем.
> Обновляется при смене задачи.

---

## Задача

**ID:** TASK-004 (pipeline done) / TASK-005 (next)
**Название:** Full training run + Evaluation framework
**Фаза:** Phase 2 - Model Training & Evaluation
**Приоритет:** High
**Статус:** Training pipeline DONE, needs GPU run

---

## Описание

Pipeline обучения полностью готов. Нужен полный прогон на GPU и evaluation framework.

---

## Чек-лист выполнения

### TASK-004: Training Pipeline [DONE]
- [x] MONAI 3D U-Net model (4.7M params)
- [x] Loss functions: DiceLoss, FocalLoss, DiceFocalLoss
- [x] Model factory (create_model, create_loss)
- [x] Trainer with train/val loop, Dice metric, callbacks
- [x] Callbacks: CheckpointCallback, EarlyStoppingCallback
- [x] Training script with CLI args + YAML config
- [x] DivisiblePadd(k=16) in transforms
- [x] Smoke test: 1 epoch on CPU passed

### Full training run
- [ ] GPU training on fold_0 (100 epochs)
- [ ] Evaluate baseline Dice score
- [ ] MLflow logging (optional enhancement)

### TASK-005: Evaluation Framework
- [ ] Per-subject Dice, IoU, sensitivity, specificity
- [ ] Stratified evaluation by lesion size
- [ ] Evaluation script
- [ ] Results visualization

---

## Связанные файлы

- Model: `src/models/unet3d.py`
- Losses: `src/models/losses.py`
- Factory: `src/models/factory.py`
- Trainer: `src/train/trainer.py`
- Callbacks: `src/train/callbacks.py`
- Train script: `scripts/train.py`
- Config: `configs/default.yaml`, `configs/experiment/baseline.yaml`
- Transforms: `src/data/transforms.py`
- Checkpoint: `outputs/fold_0/checkpoints/best_model.pth`

---

## Завершённые задачи

### TASK-004: Training Pipeline DONE
**Завершено:** 2026-02-23
- MONAI 3D U-Net, DiceFocalLoss, Trainer, callbacks, CLI script
- Smoke test: 1 epoch CPU, val_dice=0.027, checkpoint saved

### TASK-003: Preprocessing Pipeline DONE
**Завершено:** 2026-02-21

### TASK-002: Данные + EDA DONE
**Завершено:** 2026-02-19

### TASK-001: Scaffolding DONE
**Завершено:** 2026-02-19

---

**Создано:** 2026-02-21
**Последнее обновление:** 2026-02-23
