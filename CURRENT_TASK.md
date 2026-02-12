# Текущая задача

> Этот файл содержит детали задачи, над которой сейчас работаем.
> Обновляется при смене задачи.

---

## Задача

**ID:** TASK-002
**Название:** Скачать ISLES'22 и провести EDA
**Фаза:** Phase 0 - Explore
**Приоритет:** High
**Статус:** ⬜ Not Started

---

## Описание

Скачать датасет ISLES 2022 (Zenodo), валидировать файлы, провести EDA, определить splits.

---

## Чек-лист выполнения

### Скачивание данных
- [ ] Скачать ISLES 2022 с Zenodo (https://zenodo.org/record/7153326)
- [ ] Распаковать в `data/raw/isles22/`
- [ ] Проверить целостность: все кейсы, модальности (DWI, ADC, FLAIR), маски

### EDA (Exploratory Data Analysis)
- [ ] Создать `notebooks/01_eda_isles22.ipynb`
- [ ] Сколько кейсов всего
- [ ] Распределение объёмов очагов (гистограмма)
- [ ] Сколько кейсов < 1 ml (мелкие лакуны)
- [ ] Сколько posterior fossa
- [ ] Сколько мультифокальных
- [ ] Spacing/размеры volumes (мин/макс/медиана)
- [ ] Визуализация нескольких кейсов (DWI, ADC, FLAIR, маска)
- [ ] Качество: есть ли явные артефакты?

### Splits
- [ ] Определить 5-fold CV splits
- [ ] Стратификация по объёму очага
- [ ] Сохранить в `data/splits/fold_0.json` ... `fold_4.json`

### Preprocessing smoke test
- [ ] Загрузить 1 кейс через `src/data/isles22_dataset.py`
- [ ] Проверить orientation, spacing, data types

---

## Связанные файлы

- План: раздел 5, Phase 0 (Explore)
- Dataset: `src/data/isles22_dataset.py`
- Notebook: `notebooks/01_eda_isles22.ipynb`
- Splits: `data/splits/`

---

## Заметки

- ISLES'22: ~1.7 GB, NIfTI, BIDS формат
- 250 train кейсов, test скрыт
- Multi-center, multi-vendor — хороший тест на generalization
- Zenodo URL: https://zenodo.org/record/7153326

---

## Завершённые задачи

### TASK-001: Создание структуры репозитория ✅
**Завершено:** 2026-02-12 (Session 2)
- 86 файлов создано (55 Python, 6 YAML, docs, CI, Docker)
- Полная Pydantic schema (V1 + V2)
- Git инициализирован

---

## Следующая задача

После завершения этой задачи:
- **TASK-003:** Реализовать preprocessing pipeline (Phase 1)

---

**Создано:** 2026-02-12
**Последнее обновление:** 2026-02-12
