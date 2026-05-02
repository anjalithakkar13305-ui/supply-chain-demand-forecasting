# Contributing Guide

Thanks for your interest in contributing. Here's how to do it without breaking things.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/supply-chain-demand-forecasting.git
cd supply-chain-demand-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest httpx
```

---

## Before submitting a PR

**1. Run the tests**
```bash
pytest tests/ -v
```
All 48 tests must pass. If you add new functionality, add tests for it.

**2. Check your code runs end to end**
```bash
python src/data_loader.py
python src/train_model.py
python src/evaluate_model.py
```

**3. Keep commits clean**
One logical change per commit. Write clear messages:
```
feat: add LSTM model to train_model.py
fix: handle missing product names in API forecast endpoint
docs: update README with new model results
test: add unit tests for database.py queries
```

---

## Where to add things

| What | Where |
|---|---|
| New ML model | `src/train_model.py` — add alongside XGBoost |
| New API endpoint | `api/main.py` — follow existing pattern, use `/v1/` prefix |
| New dashboard tab | `dashboard/app.py` — add to `st.tabs()` list |
| New SQL query | `src/database.py` — follow existing function pattern |
| New features | `src/feature_engineering.py` — add to `build_feature_matrix()` |

---

## What not to do

- Don't commit `data/supply_chain.csv`, `*.db`, or `*.pkl` files
- Don't break the existing test suite
- Don't change the API response schema without updating the Pydantic models
