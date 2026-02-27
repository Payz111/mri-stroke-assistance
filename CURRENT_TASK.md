# Текущая задача

> Этот файл содержит детали задачи, над которой сейчас работаем.
> Обновляется при смене задачи.

---

## Задача

**ID:** TASK-006
**Название:** Structured Findings + Report Generation
**Фаза:** Phase 2 - Model Training & Evaluation
**Приоритет:** High
**Статус:** DONE

---

## Описание

Полный pipeline: сегментационная маска -> структурированные findings (JSON) -> текстовый радиологический отчёт.
Принцип: ZERO hallucinations -- каждое предложение отчёта детерминистически из JSON.

---

## Чек-лист выполнения

### TASK-006: Structured Findings + Report Generation [DONE]
- [x] volume_calc.py -- объём в mL + max diameter (bbox diagonal)
- [x] laterality.py -- определение полушария (left/right/bilateral)
- [x] location.py -- эвристическая анатомическая локализация (centroid-based)
- [x] territory.py -- сосудистый бассейн (MCA/ACA/PCA/vertebrobasilar)
- [x] mismatch.py -- DWI-FLAIR mismatch (acute vs subacute)
- [x] builder.py -- сборка V1Findings JSON (все модули + ADC/FLAIR analysis)
- [x] templates.py -- шаблоны отчёта (ZERO hallucinations)
- [x] generator.py -- генератор текста из JSON
- [x] validator.py -- валидация отчёта vs findings (5 проверок)
- [x] Smoke test: синтетические данные -- PASSED

---

## Связанные файлы

### Findings extraction
- Volume calc: `src/findings/volume_calc.py`
- Laterality: `src/findings/laterality.py`
- Location: `src/findings/location.py`
- Territory: `src/findings/territory.py`
- Mismatch: `src/findings/mismatch.py`
- Builder: `src/findings/builder.py`
- Schema: `src/findings/schema.py`

### Report generation
- Templates: `src/report/templates.py`
- Generator: `src/report/generator.py`
- Validator: `src/report/validator.py`

### Testing
- Smoke test: `scripts/smoke_test_findings.py`
- Outputs: `outputs/smoke_test_findings/`

---

## Следующие задачи

1. TASK-007: Run full evaluation with best_model.pth on val set
2. Demo app update (Gradio с загрузкой NIfTI + отчёт)
3. Documentation + portfolio polish

---

## Завершённые задачи

### TASK-006: Structured Findings + Report DONE
**Завершено:** 2026-02-27

### TASK-005: Evaluation Framework DONE
**Завершено:** 2026-02-23

### TASK-004: Training Pipeline DONE
**Завершено:** 2026-02-23
- Best val_dice: 0.606 (epoch 93, 100 epochs on Kaggle T4)

### TASK-003: Preprocessing Pipeline DONE
**Завершено:** 2026-02-21

### TASK-002: Данные + EDA DONE
**Завершено:** 2026-02-19

### TASK-001: Scaffolding DONE
**Завершено:** 2026-02-19

---

**Создано:** 2026-02-21
**Последнее обновление:** 2026-02-27
