# ── Supply Chain Demand Forecasting — Makefile ────────────────────────────────
# Usage: make <target>
# Run `make help` to see all available commands.

.PHONY: help setup data train evaluate api dashboard docker-build docker-up test clean

PYTHON = python
PIP    = pip

help:
	@echo ""
	@echo "  Supply Chain Demand Forecasting"
	@echo "  ================================"
	@echo ""
	@echo "  Setup & Data"
	@echo "    make setup       Install all Python dependencies"
	@echo "    make data        Load CSV into SQLite database"
	@echo ""
	@echo "  Model"
	@echo "    make train       Train XGBoost + Prophet, save best model"
	@echo "    make evaluate    Run evaluation, generate diagnostic plots"
	@echo ""
	@echo "  Run"
	@echo "    make api         Start FastAPI server on port 8000"
	@echo "    make dashboard   Launch Streamlit dashboard on port 8501"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build  Build Docker images"
	@echo "    make docker-up     Start API + Dashboard via docker-compose"
	@echo "    make docker-down   Stop all containers"
	@echo ""
	@echo "  Tests"
	@echo "    make test        Run all unit tests"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean       Remove generated files (DB, models, reports)"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed."

# ── Data ──────────────────────────────────────────────────────────────────────
data:
	$(PYTHON) src/data_loader.py
	@echo "✅ Data loaded into SQLite."

# ── Model ─────────────────────────────────────────────────────────────────────
train:
	$(PYTHON) src/train_model.py
	@echo "✅ Models trained and saved."

evaluate:
	$(PYTHON) src/evaluate_model.py
	@echo "✅ Evaluation complete. Check reports/ for plots."

# ── Run ───────────────────────────────────────────────────────────────────────
api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f data/supply_chain.db
	rm -f models/*.pkl models/*.json
	rm -f reports/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned generated files."
