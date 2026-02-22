# Текущая задача

> Этот файл содержит детали задачи, над которой сейчас работаем.
> Обновляется при смене задачи.

---

## Задача

**ID:** TASK-003
**Название:** Реализовать preprocessing pipeline
**Фаза:** Phase 1 - Data & Preprocessing
**Приоритет:** High
**Статус:** ⬜ Not Started

---

## Описание

Реализовать полный preprocessing pipeline для подготовки MRI данных к обучению модели.

---

## Чек-лист выполнения

### Dataset class
- [ ] Реализовать `src/data/isles22_dataset.py`
- [ ] Метод `__getitem__` возвращает dict: {dwi, adc, flair, mask, metadata}
- [ ] Загрузка splits из JSON
- [ ] Smoke test: загрузить 1 кейс

### Preprocessing components
- [ ] `src/preprocess/registration.py` — co-registration ADC/FLAIR → DWI
- [ ] `src/preprocess/intensity_norm.py` — z-score normalization
- [ ] `src/data/transforms.py` — MONAI transforms (train/val)
- [ ] `src/preprocess/pipeline.py` — end-to-end pipeline

### Smoke test
- [ ] Загрузить 1 кейс из fold_0 train
- [ ] Применить preprocessing
- [ ] Проверить shapes, data types, ranges
- [ ] Визуализация: до/после preprocessing

---

## Связанные файлы

- План: раздел 5.2, Phase 1
- Dataset: `src/data/isles22_dataset.py`
- Preprocessing: `src/preprocess/`
- Transforms: `src/data/transforms.py`
- Splits: `data/splits/fold_0.json`

---

## Заметки

**Технические требования:**
- Target spacing: 1x1x1 mm или 2x2x2 mm (определить)
- Orientation: RAS
- Normalization: z-score per modality
- Registration: rigid, moving=ADC/FLAIR, fixed=DWI

---

## Завершённые задачи

### TASK-002: Данные + EDA ✅ ЗАВЕРШЕНО
**Завершено:** 2026-02-19
- Датасет загружен (250 кейсов)
- EDA + метаданные собраны
- **5-fold CV splits созданы** (200 train / 50 val, стратификация)

### TASK-001: Scaffolding ✅ ЗАВЕРШЕНО
**Завершено:** 2026-02-19
- 86 файлов, структура готова

---

## Следующая задача

После завершения:
- **TASK-004:** Baseline 3D U-Net training (Phase 2)

---

**Создано:** 2026-02-19
**Последнее обновление:** 2026-02-19
