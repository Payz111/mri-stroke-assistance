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
| Gate A - Feasibility | 🟡 IN PROGRESS | 15% | Репо создано, ждём данные + EDA |
| Gate B - Baseline Training | ⬜ NOT STARTED | 0% | |
| Gate C - Evaluation | ⬜ NOT STARTED | 0% | |
| Gate D - Structured Findings | ⬜ NOT STARTED | 0% | Schema готова (Pydantic) |
| Gate E - Report Generation | ⬜ NOT STARTED | 0% | |
| Gate F - Demo & Docs | ⬜ NOT STARTED | 5% | Gradio skeleton, README, Dockerfile |
| Gate G - V2 Perfusion | ⬜ NOT STARTED | 0% | |
| Gate H - Portfolio Ready | ⬜ NOT STARTED | 0% | |

**Легенда:** ⬜ Not Started | 🟡 In Progress | ✅ Complete | 🔴 Blocked

---

## Текущие задачи (This Session)

### В работе сейчас
- [ ] TASK-002: Скачать ISLES'22 и валидировать данные

### Следующие задачи
1. EDA датасета (notebooks/01_eda_isles22.ipynb)
2. Определить splits (5-fold CV)
3. Реализовать preprocessing pipeline

---

## Выполнено (история)

### Session 2 (2026-02-12) — Claude Code
- [x] **TASK-001: Создание структуры репозитория** — ПОЛНОСТЬЮ ЗАВЕРШЕНО
  - Создано 86 файлов, 55 Python модулей
  - pyproject.toml (PyTorch, MONAI, nibabel, SimpleITK, Hydra, MLflow, Gradio)
  - Makefile (setup, train, eval, demo, docker, test, lint)
  - .gitignore, Dockerfile, LICENSE (MIT)
  - GitHub Actions CI (lint + test)
  - 5 YAML конфигов (default, baseline, augmented, metrics, v2_ctp)
  - Полная Pydantic schema (V1 + V2) в src/findings/schema.py
  - Все skeleton модули с stub functions и type hints
  - 5 тестовых файлов (schema, QC, preprocessing, inference, report)
  - Gradio demo skeleton (demo/app.py)
  - 4 docs (PRD, Architecture, Model Card, Evaluation Report)
  - Git инициализирован

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
| `PROJECT_STATE.md` | Этот файл — состояние проекта | ✅ |
| `DECISIONS.md` | Архитектурные решения (10 ADR) | ✅ |
| `CURRENT_TASK.md` | Детали текущей задачи | ✅ |
| `SESSION_START.md` | Протокол начала сессии | ✅ |
| `src/` | Исходный код (skeleton) | ✅ 42 файла |
| `src/findings/schema.py` | Pydantic V1+V2 schema | ✅ Готова |
| `configs/` | YAML конфигурации | ✅ 5 файлов |
| `tests/` | Тесты (skeleton + schema) | ✅ 5 файлов |
| `demo/app.py` | Gradio demo | ✅ Skeleton |
| `data/` | Датасеты | ⬜ Не загружено |

---

## Известные проблемы / Blockers

(пока нет)

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

**Последнее обновление:** 2026-02-12
**Обновил:** Claude Code (Session 2)
