.PHONY: setup data-isles22 data-isles24 preprocess train train-fold eval infer demo docker-build docker-run test lint clean

# Setup
setup:
	pip install -e ".[dev]"
	pre-commit install

# Data
data-isles22:
	bash scripts/download_isles22.sh

data-isles24:
	bash scripts/download_isles24.sh

preprocess:
	python scripts/preprocess_dataset.py --config configs/default.yaml

# Training
train:
	python scripts/train.py --config-name baseline

train-fold:
	python scripts/train.py --config-name baseline training.fold=$(FOLD)

# Evaluation
eval:
	python scripts/evaluate.py --config-name default

# Inference
infer:
	python scripts/infer_single.py --input $(INPUT) --output $(OUTPUT)

# Demo
demo:
	python demo/app.py

# Docker
docker-build:
	docker build -t mri-stroke-assist:latest .

docker-run:
	docker run -p 7860:7860 mri-stroke-assist:latest

# Development
test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

format:
	ruff check --fix src/ scripts/ tests/
	black src/ scripts/ tests/

# Clean
clean:
	rm -rf outputs/predictions/*
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
