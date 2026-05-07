# Financial Operations Analytics System

End-to-end Python analytics: **weekly revenue forecasting** (Holt–Winters, SARIMA, Prophet), **churn** (logistic regression, random forest, gradient boosting, XGBoost, LightGBM with CV tuning), **survival analysis** (Kaplan–Meier, Cox PH), **RFM + K-Means**, **cohort retention**, **profit-driver regression**, **Monte Carlo** on orders, **BI exports** (CSV/Parquet), and a **static HTML executive dashboard**.

## Quick start

```bash
cd "FINANCIAL ANALYTICAL SYSTEM"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
python run_pipeline.py
```

Outputs land in `artifacts/` (metrics JSON, figures, tables, Parquet/CSV, `dashboard/index.html`).

## Project layout

| Path | Purpose |
|------|---------|
| `src/fo_analytics/` | Library code (data, forecasting, churn, survival, segmentation, export, dashboard) |
| `run_pipeline.py` | One-shot end-to-end run |
| `artifacts/` | Generated outputs (gitignored) |
| `docs/model_cards/` | Model documentation |
| `docs/schema.md` | BI data dictionary |
| `docs/USER_TESTING.md` | Expert review template |
| `reports/ACADEMIC_REPORT.md` | IMRAD-style report source (expand to ~22 pages with figures) |
| `api/` | Optional FastAPI app (`pip install -e ".[api]"` then `uvicorn api.main:app --reload`) |

## Reproducibility

- Fixed random seeds in `fo_analytics.config`.
- `requirements.txt` pins versions; adjust if Prophet/LightGBM fail to build on your OS.

## Evaluation standards

- Time-series: train on history, hold out the last **26 weeks** for error (MAPE, sMAPE, RMSE) and information criteria where applicable.
- Churn: **stratified** train/test split; optional **time-based** cutoff documented in metrics.
- No future fields in churn features: all signals computed with **observation-date** cutoff.

## Definition of done

Baseline modules + extensions A–H are implemented or marked N/A in `docs/NON_APPLICABLE.md`. API docs apply only if you run the FastAPI service.
