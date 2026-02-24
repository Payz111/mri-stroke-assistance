# MRI Stroke Assist — Состояние проекта

> **ВАЖНО**: Этот файл — "память" проекта. Claude читает его в начале каждой сессии.
> Обновляется после каждой рабочей сессии.

---

## Текущая фаза

```
Phase: 0 - EXPLORE (в процессе)
Sprint: Week 1
Status: IN PROGRESS
```

## Прогресс по Gate'ам

| Gate | Статус | Завершено | Заметки |
|------|--------|-----------|---------|
| Gate A - Feasibility | ✅ COMPLETE | 100% | Данные + EDA + splits готовы |
| Gate B - Baseline Training | 🟡 IN PROGRESS | 85% | Pipeline done, GPU training on Kaggle |
| Gate C - Evaluation | 🟡 IN PROGRESS | 80% | Metrics + stratified eval + error analysis done |
| Gate D - Structured Findings | ⬜ NOT STARTED | 0% | Schema готова (Pydantic) |
| Gate E - Report Generation | ⬜ NOT STARTED | 0% | |
| Gate F - Demo & Docs | ⬜ NOT STARTED | 5% | Gradio skeleton, README, Dockerfile |
| Gate G - V2 Perfusion | ⬜ NOT STARTED | 0% | |
| Gate H - Portfolio Ready | ⬜ NOT STARTED | 0% | |

**Легенда:** ⬜ Not Started | 🟡 In Progress | ✅ Complete | 🔴 Blocked

---

## Текущие задачи (This Session)

### В работе сейчас
- [x] TASK-005: Evaluation framework -- DONE
- [ ] GPU training running on Kaggle (fold_0, 100 epochs)

### Следующие задачи
1. Wait for Kaggle training, run evaluate.py on best checkpoint
2. TASK-006: Structured findings + report generation

---

## Выполнено (история)

### Session 4 (2026-02-23) -- Claude Code, TASK-004 + TASK-005

**TASK-004: Baseline 3D U-Net Training Pipeline DONE**
- Model: MONAI 3D U-Net (4.7M params), configurable features/dropout
- Loss: DiceFocalLoss (Dice + Focal combined), DiceLoss, FocalLoss
- Factory: create_model() + create_loss() from config dicts
- Callbacks: CheckpointCallback (best model), EarlyStoppingCallback
- Trainer: full train/val loop with Dice metric, LR scheduler, logging
- Training script: scripts/train.py (argparse CLI, YAML config loading)
- DivisiblePadd(k=16) added to transforms (fixes UNet dim mismatch on odd sizes)
- Smoke test: 1 epoch on CPU passed (val_dice=0.027, checkpoint saved)
- Next: full GPU training run for real baseline Dice score

**TASK-005: Evaluation Framework DONE**
- Metrics: Dice, IoU, sensitivity, specificity, HD95, volume MAE, lesion-wise F1
- Stratified evaluation by lesion size (tiny/small/medium/large)
- Error analysis: FP/FN categorization, missed/spurious lesions
- Evaluation CLI script (scripts/evaluate.py)
- Kaggle training notebook created and launched
- Unit tests passed on synthetic data

### Session 3 (2026-02-21) -- Claude Code, TASK-003

**TASK-003: Preprocessing Pipeline DONE**
- Dataset class: src/data/isles22_dataset.py (BIDS loading, __getitem__ returns dict)
- Intensity normalization: src/preprocess/intensity_norm.py (z-score, percentile)
- MONAI transforms: src/data/transforms.py (ResampleToReference, NormalizePerModality, StackModalities)
- Registration: src/preprocess/registration.py (SimpleITK rigid/affine + scipy resample fallback)
- Pipeline: src/preprocess/pipeline.py (per-subject + batch preprocessing)
- Smoke test: PASSED -- image (3,112,112,73), label (1,112,112,73), mask binary, 835 lesion voxels
- Key finding: FLAIR has different spatial resolution (281x352x352 vs DWI 112x112x73) -- ResampleToReference solves this

### Session 2 (2026-02-19) — Claude Code, TASK-001 + TASK-002 (частично)

**TASK-001: Scaffolding ✅ ЗАВЕРШЕНО**
- Создано 86 файлов, 55 Python модулей
- pyproject.toml (PyTorch, MONAI, nibabel, SimpleITK, Hydra, MLflow, Gradio)
- Makefile, .gitignore, Dockerfile, LICENSE (MIT)
- GitHub Actions CI, 5 YAML конфигов
- Полная Pydantic schema (V1 + V2) в src/findings/schema.py
- Все skeleton модули, 5 тестовых файлов
- Gradio demo skeleton, 4 docs
- Git init + remote GitLab, initial commit + push

**TASK-002: Данные + EDA ✅ ЗАВЕРШЕНО**
- ✅ ISLES 2022 скачан и распакован (~1.7 GB, 250 кейсов)
- ✅ EDA ноутбук создан (notebooks/01_eda_isles22.ipynb)
- ✅ Метаданные собраны для всех 250 кейсов (data/processed/isles22_metadata.csv)
- ✅ Статистика: медиана 6.66 ml, диапазон 0-482 ml
- ✅ Стратификация: 43 tiny (<1ml), 95 small, 79 medium, 30 large
- ✅ Spacing проверен: большинство 2x2x2 мм, есть variability
- ⚠️ **Найдено 3 кейса с пустыми масками** (0 voxels) — включены в splits
- ✅ **5-fold CV splits созданы** (200 train / 50 val, стратификация по размеру)

**Технические проблемы и решения:**
1. **pyproject.toml fix:** Добавлен `[tool.hatch.build.targets.wheel] packages = ["src"]` для hatchling
2. **Зависимости:** jupyter, ipykernel, tqdm добавлены в dev dependencies
3. **Python версия:** Используется Python 3.14, packages установлены корректно
4. **Кодировка:** Windows cp1251 — избегать unicode символов в выводе

### Session 1 (2026-02-04) — Claude Web
- [x] Обсуждение концепции проекта
- [x] Создание PRD v1.0
- [x] Создание полного плана разработки (mri_stroke_assist_plan.md)
- [x] Создание системы управления контекстом (PROJECT_STATE, CURRENT_TASK, DECISIONS, SESSION_START)

---

## Технические решения (зафиксированы)

| Решение | Выбор | Причина |
|---------|-------|---------|
| Framework | PyTorch + MONAI | Индустриальный стандарт medical imaging |
| Segmentation | 3D U-Net (baseline) | Простота, хорошие результаты |
| Config | Hydra + OmegaConf | Reproducibility |
| Tracking | MLflow | Self-hosted, бесплатно |
| Demo | Gradio | Быстро, просто |
| Report | Template-based + validator | 0 галлюцинаций |
| QC policy | No-score на critical fail | Safety-first |
| Masks | NIfTI (.nii.gz) | Medical imaging стандарт |
| Findings | JSON (Pydantic) | Типизированный, валидируемый |

---

## Ключевые файлы проекта

| Файл | Описание | Статус |
|------|----------|--------|
| `PROJECT_STATE.md` | Этот файл — состояние проекта | ✅ Обновлён |
| `DECISIONS.md` | Архитектурные решения (10 ADR) | ✅ |
| `CURRENT_TASK.md` | Детали текущей задачи | ✅ Обновлён |
| `SESSION_START.md` | Протокол начала сессии | ✅ |
| `src/` | Исходный код (skeleton) | ✅ 42 файла |
| `src/findings/schema.py` | Pydantic V1+V2 schema | ✅ Готова |
| `configs/` | YAML конфигурации | ✅ 5 файлов |
| `tests/` | Тесты (skeleton + schema) | ✅ 5 файлов |
| `demo/app.py` | Gradio demo | ✅ Skeleton |
| `data/raw/isles22/` | ISLES 2022 датасет | ✅ 250 кейсов |
| `data/processed/isles22_metadata.csv` | Метаданные датасета | ✅ Собраны |
| `notebooks/01_eda_isles22.ipynb` | EDA ноутбук | ✅ Создан и выполнен |
| `data/splits/` | 5-fold CV splits | ✅ 5 folds созданы |
| `scripts/create_splits.py` | Скрипт создания splits | ✅ Готов |

---

## Известные проблемы / Blockers

### 1. Пустые маски (3 кейса)
- **Кейсы:** sub-strokecase0150, sub-strokecase0151, sub-strokecase0170
- **Проблема:** Маски содержат 0 voxels
- **Возможные причины:** отрицательные контроли, ошибки аннотации, очень мелкие очаги
- **Решение:** Включить в обучение, но отследить отдельно в evaluation

### 2. Variability в spacing
- **Проблема:** Spacing варьируется от 0.88 до 2.0 мм (in-plane), 2.0-5.0 мм (slice thickness)
- **Решение:** Preprocessing с resampling к единому spacing (1x1x1 мм или 2x2x2 мм)

### 3. Python версии
- **Проблема:** В системе установлены Python 3.11 и 3.14, packages в 3.14
- **Решение:** Использовать явно Python 3.14 для всех команд

---

## Важные заметки для Claude

1. **Владелец проекта**: Ринат — невролог, хочет войти в Data Science
2. **Цель**: Портфолио-проект для поиска работы в Medical Imaging ML
3. **Особенность**: Ринат сам эксперт в чтении МРТ мозга — это уникальное преимущество
4. **V1**: DWI/ADC/FLAIR → сегментация + findings + report
5. **V2**: + CTP perfusion → core/penumbra/mismatch
6. **Главный принцип**: 0 галлюцинаций — текст только из structured JSON
7. **Рабочая среда**: Windows 11, Claude Code (CLI), VS Code
8. **Репозиторий**: `c:\Users\mfayz\MRI Stroke Assistance`

---

## Протокол начала сессии

Когда пользователь начинает новую сессию разработки, Claude должен:

1. Прочитать `PROJECT_STATE.md` (этот файл)
2. Прочитать `CURRENT_TASK.md` для деталей текущей задачи
3. Проверить `DECISIONS.md` если нужен контекст решений
4. Спросить: "Продолжаем с [текущая задача]?" или предложить следующий шаг

---

**Последнее обновление:** 2026-02-23
**Обновил:** Claude Code (Session 4)
