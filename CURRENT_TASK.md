# Текущая задача

> Этот файл содержит детали задачи, над которой сейчас работаем.
> Обновляется при смене задачи.

---

## Задача

**ID:** TASK-004
**Название:** Baseline 3D U-Net training
**Фаза:** Phase 2 - Model Training
**Приоритет:** High
**Статус:** Not Started

---

## Описание

Обучить baseline 3D U-Net модель сегментации инфарктного очага.

---

## Чек-лист выполнения

### Модель
- [ ] Реализовать `src/models/unet3d.py` (MONAI 3D U-Net)
- [ ] Конфигурация модели через Hydra config
- [ ] Loss function: Dice + BCE

### Training loop
- [ ] `src/training/trainer.py` -- основной training loop
- [ ] Metrics: Dice score, IoU, sensitivity, specificity
- [ ] Checkpointing (best model by val Dice)
- [ ] MLflow logging

### Запуск обучения
- [ ] Обучение на fold_0 (200 train / 50 val)
- [ ] Мониторинг loss и Dice на val
- [ ] Оценка baseline Dice score

---

## Связанные файлы

- Model: `src/models/unet3d.py`
- Trainer: `src/training/trainer.py`
- Config: `configs/experiment/baseline.yaml`
- Dataset: `src/data/isles22_dataset.py`
- Transforms: `src/data/transforms.py`

---

## Завершённые задачи

### TASK-003: Preprocessing Pipeline DONE
**Завершено:** 2026-02-21
- Dataset class реализован (isles22_dataset.py)
- Intensity normalization (z-score, percentile)
- MONAI transforms (ResampleToReference, NormalizePerModality, StackModalities)
- Registration (SimpleITK rigid + scipy resample)
- End-to-end pipeline (per-subject + batch)
- Smoke test PASSED

### TASK-002: Данные + EDA DONE
**Завершено:** 2026-02-19
- Датасет загружен (250 кейсов)
- EDA + метаданные собраны
- **5-fold CV splits созданы** (200 train / 50 val, стратификация)

### TASK-001: Scaffolding DONE
**Завершено:** 2026-02-19
- 86 файлов, структура готова

---

## Следующая задача

После завершения:
- **TASK-005:** Evaluation framework (Phase 2)

---

**Создано:** 2026-02-21
**Последнее обновление:** 2026-02-21
