# AML Transaction Monitoring

Leak-free anti-money-laundering transaction monitoring on ~5M bank transfers (IBM AML, HI-Small):
classical ML, statistical evaluation, MLflow experiment tracking, a FastAPI scoring API and a Streamlit alert dashboard.

> 🚧 Work in progress. Results, demo link and full write-up are added as each step completes (see `AML_STEPS.md`).

## Quick start
```bash
uv sync            # create .venv and install pinned dependencies
uv run pytest      # run tests
```

## Project layout
```
configs/     schema mapping (works with any transactions CSV) + training settings
notebooks/   Kaggle notebooks for the heavy steps (EDA, feature building)
src/aml/     pipeline code: data, features, models, evaluation, explanations
api/         FastAPI scoring service
app/         Streamlit dashboard
tests/       leakage, schema and metric tests
reports/     auto-generated results (the only source of reported numbers)
data/        local data (git-ignored)
```
