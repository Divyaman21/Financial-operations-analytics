# N/A items (extensions H subset)

| Item | Status | Notes |
|------|--------|--------|
| FastAPI / OpenAPI | Optional | Install `pip install -e ".[api]"` and run `uvicorn api.main:app`. If you do not deploy an API, API docs are **N/A**. |
| Live user testing | Template only | Use `docs/USER_TESTING.md` to record real sessions; synthetic runs do not replace domain review. |
| AUC-ROC > 0.75 | Data-dependent | Document achieved metrics in `artifacts/metrics/churn.json` and explain class overlap / label noise if below target. |
