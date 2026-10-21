# FraudShield Makefile (Convenience wrapper around python -m fraudshield)
PYTHON ?= python

.PHONY: help data features train evaluate test serve clean

help:
	@echo "Available commands:"
	@echo "  make data      - Ingest and verify raw data"
	@echo "  make features  - Build feature sets A, B, and C"
	@echo "  make train     - Train baseline and XGBoost models"
	@echo "  make evaluate  - Run evaluation, sweeps, and experiments"
	@echo "  make test      - Run test suite"
	@echo "  make serve     - Run FastAPI app and dashboard"

data:
	$(PYTHON) -m fraudshield data

features:
	$(PYTHON) -m fraudshield features

train:
	$(PYTHON) -m fraudshield train

evaluate:
	$(PYTHON) -m fraudshield evaluate

test:
	$(PYTHON) -m fraudshield test

serve:
	$(PYTHON) -m fraudshield serve

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
