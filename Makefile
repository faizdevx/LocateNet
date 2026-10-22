# Usage: make setup | init | run | clean   (Windows without make: use scripts/dev.ps1)
PYTHON ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup init run clean reset-data eval-data eval-all eval-report eval-regress eval-regress-smoke

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Edit .env and set LOCATENET_SECRET_KEY, then run: make init"

init:
	$(BIN)/python init_system.py

run:
	$(BIN)/python app.py

# Remove caches only; your data is untouched.
clean:
	find . -name __pycache__ -type d -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

# DESTRUCTIVE: deletes the database, uploads, resources, logs and the vector index.
reset-data:
	@read -p "This deletes ALL local data. Type 'yes' to continue: " ans && [ "$$ans" = "yes" ]
	$(BIN)/python clean_system.py

# --- Evaluation (Week 2). Datasets go to data/ and caches to eval/cache/ (both git-ignored). ---
EVAL_ORDER = embed_lfw verification openset openset_nomerge operating_point calibration bias robustness quality detection detsize_check latency label_noise

eval-data:
	$(BIN)/pip install -r requirements-eval.txt
	$(BIN)/python scripts/get_eval_data.py

eval-all:
	for exp in $(EVAL_ORDER); do $(BIN)/python -m eval run eval/config/$$exp.yaml || exit 1; done
	$(BIN)/python -m eval report

eval-report:
	$(BIN)/python -m eval report

# Regression gate: re-run the headline suite and compare with eval/results/baseline (fails on regressions)
eval-regress:
	$(BIN)/python -m eval regress

eval-regress-smoke:
	$(BIN)/python -m eval regress --smoke
