# Financial Operations Analytics System

End-to-end Python analytics: **weekly revenue forecasting** (Holt–Winters, SARIMA, Prophet), **churn** (logistic regression, random forest, gradient boosting, XGBoost, LightGBM with CV tuning), **survival analysis** (Kaplan–Meier, Cox PH), **RFM + K-Means**, **cohort retention**, **profit-driver regression**, **Monte Carlo** on orders, **BI exports** (CSV/Parquet), and a **static HTML executive dashboard**.

## Quick start

```bash
cd "Financial Analytical System"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e ".[api]"
python3 run_pipeline.py
open artifacts/dashboard/index.html
uvicorn api.main:app --reload &
sleep 2
open http://127.0.0.1:8000/docs
```

Outputs land in `artifacts/` (metrics JSON, figures, tables, Parquet/CSV, `dashboard/index.html`). On Windows, you can use `start artifacts/dashboard/index.html` or `explorer artifacts/dashboard/index.html`.

## API Testing & Sample Payloads

Once the API is running, open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to access the Swagger UI. Click **"Try it out"** and use these sample values:

**1. `POST /api/v1/churn/predict` (Real Churn Model)**
```json
{
  "customer_id": 1042,
  "recency_days": 15.5,
  "frequency_365d": 12.0,
  "monetary_365d": 1450.75,
  "tenure_days": 450.0,
  "avg_interpurchase_days": 25.5,
  "clv_proxy": 3200.50
}
```

**2. `GET /api/v1/forecast/26-weeks` (Revenue Forecast)**
- **`scenario`**: Enter `base`, `optimistic`, or `pessimistic`.

**3. `POST /score/churn_stub` (Heuristic Fallback)**
```json
{
  "customer_id": 2084,
  "recency_days": 85.0,
  "frequency_365d": 2.0,
  "monetary_365d": 120.0,
  "tenure_days": 90.0,
  "avg_interpurchase_days": 45.0,
  "clv_proxy": 240.0
}
```
For all other endpoints (`/api/v1/health`, `/api/v1/models`, `/api/v1/drift/latest`, etc.), no inputs are required. Simply click **"Execute"**.

## Project layout

| Path | Purpose |
|------|---------|
| `src/fo_analytics/` | Library code (data, forecasting, churn, survival, segmentation, export, dashboard) |
| `run_pipeline.py` | One-shot end-to-end run |
| `artifacts/` | Generated outputs (gitignored) |
| `docs/` | Comprehensive documentation (Architecture, API, Docker, schema, etc.) |
| `docs/model_cards/` | Model documentation |
| `reports/ACADEMIC_REPORT.md` | IMRAD-style report source (expand to ~22 pages with figures) |
| `api/` | Optional FastAPI app (`pip install -e ".[api]"` then `uvicorn api.main:app --reload`) |
| `tests/` | Unit and integration tests |
| `Dockerfile` & `docker-compose.yml` | Containerization configurations |

## Reproducibility

- Fixed random seeds in `fo_analytics.config`.
- `requirements.txt` pins versions; adjust if Prophet/LightGBM fail to build on your OS.

## Evaluation standards

- Time-series: train on history, hold out the last **26 weeks** for error (MAPE, sMAPE, RMSE) and information criteria where applicable.
- Churn: **stratified** train/test split; optional **time-based** cutoff documented in metrics.
- No future fields in churn features: all signals computed with **observation-date** cutoff.

## Definition of done

Baseline modules + extensions A–H are implemented or marked N/A in `docs/NON_APPLICABLE.md`. API docs apply only if you run the FastAPI service.
