# Usage: make setup | init | run | clean   (Windows without make: use scripts/dev.ps1)
PYTHON ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: setup init run clean reset-data

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
