# Текущая задача

> Этот файл содержит детали задачи, над которой сейчас работаем.
> Обновляется при смене задачи.

---

## Задача

**ID:** TASK-002
**Название:** Скачать ISLES'22 и провести EDA
**Фаза:** Phase 0 - Explore
**Приоритет:** High
**Статус:** 🟡 In Progress (70% complete)

---

## Описание

Скачать датасет ISLES 2022 (Zenodo), валидировать файлы, провести EDA, определить splits.

---

## Чек-лист выполнения

### Скачивание данных ✅ DONE
- [x] Скачать ISLES 2022 с Zenodo (https://zenodo.org/record/7153326)
- [x] Распаковать в `data/raw/isles22/`
- [x] Проверить целостность: все кейсы, модальности (DWI, ADC, FLAIR), маски
  - 250 кейсов ✓
  - Все модальности присутствуют ✓
  - Masks в derivatives/ ✓

### EDA (Exploratory Data Analysis) ✅ DONE
- [x] Создать `notebooks/01_eda_isles22.ipynb`
- [x] Сколько кейсов всего — **250**
- [x] Распределение объёмов очагов (гистограмма) — медиана 6.66 ml
- [x] Сколько кейсов < 1 ml (мелкие лакуны) — **43 кейса (17.2%)**
- [x] Сколько posterior fossa — TODO (требует анализа центра масс)
- [x] Сколько мультифокальных — TODO (требует connected components)
- [x] Spacing/размеры volumes (мин/макс/медиана)
  - In-plane: 0.88-2.0 мм (медиана 2.0)
  - Slice: 2.0-5.0 мм (медиана 2.0)
  - Shapes: 112-256 x 112-256 x 25-76
- [x] Визуализация нескольких кейсов (DWI, ADC, FLAIR, маска)
- [x] Качество: найдено 3 кейса с пустыми масками

### Splits ⏸️ TODO — СЛЕДУЮЩИЙ ШАГ
- [ ] Определить 5-fold CV splits
- [ ] Стратификация по объёму очага
- [ ] Сохранить в `data/splits/fold_0.json` ... `fold_4.json`

### Preprocessing smoke test ⏸️ TODO
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

**Общие:**
- ISLES'22: ~1.7 GB, NIfTI, BIDS формат
- 250 train кейсов, test скрыт
- Multi-center, multi-vendor — хороший тест на generalization
- Zenodo URL: https://zenodo.org/record/7153326

**Находки из EDA:**
- Медиана объёма: 6.66 ml (маленькие очаги преобладают)
- Стратификация: 43 tiny, 95 small, 79 medium, 30 large
- **3 кейса с пустыми масками:** sub-strokecase0150, 0151, 0170 (требует проверки)
- Spacing variability требует preprocessing с resampling

**Технические проблемы:**
- pyproject.toml нужен был fix для hatchling (packages = ["src"])
- Python 3.14 используется для всех зависимостей
- Windows cp1251 кодировка — избегать unicode в выводе

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
**Последнее обновление:** 2026-02-19
