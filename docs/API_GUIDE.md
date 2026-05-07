# API Reference Guide

## Base URL
```
http://localhost:8000
```

## Authentication
No authentication required for the demo API. In production, add OAuth2 or API key middleware.

---

## Endpoints

### Root
```bash
GET /
```
Returns welcome message and lists all available endpoints.

```bash
curl http://localhost:8000/
```

---

### Health Check
```bash
GET /health
```
Returns basic server health status.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

### Latest Metrics
```bash
GET /artifacts/latest_metrics
```
Returns all pipeline metrics as a JSON object.

```bash
curl http://localhost:8000/artifacts/latest_metrics
```

> **Note:** Requires `python run_pipeline.py` to have been run at least once.

---

## V1 Endpoints (Versioned API)

### Predict Churn
```bash
POST /api/v1/churn/predict
```

**Request Body:**
```json
{
  "customer_id": 42,
  "recency_days": 15.0,
  "frequency_365d": 12.0,
  "monetary_365d": 2500.0,
  "tenure_days": 400,
  "avg_interpurchase_days": 30.0,
  "clv_proxy": 850.0
}
```

**Response:**
```json
{
  "customer_id": 42,
  "churn_probability": 0.1823,
  "risk_tier": "Low",
  "model_version": "registry_latest"
}
```

**curl example:**
```bash
curl -X POST http://localhost:8000/api/v1/churn/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":42,"recency_days":15,"frequency_365d":12,"monetary_365d":2500,"tenure_days":400,"avg_interpurchase_days":30,"clv_proxy":850}'
```

---

### Revenue Forecast
```bash
GET /api/v1/forecast/26-weeks?scenario=base
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| scenario | string | base | One of: `base`, `optimistic`, `pessimistic` |

```bash
curl "http://localhost:8000/api/v1/forecast/26-weeks?scenario=optimistic"
```

---

### Data Drift Status
```bash
GET /api/v1/drift/latest
```
Returns the latest drift baseline and monitoring data.

```bash
curl http://localhost:8000/api/v1/drift/latest
```

---

### Latest Alerts
```bash
GET /api/v1/alerts/latest
```
Returns triggered alerts from the most recent pipeline run.

```bash
curl http://localhost:8000/api/v1/alerts/latest
```

---

### List Models
```bash
GET /api/v1/models
```
Returns all registered models with version history and latest metrics.

```bash
curl http://localhost:8000/api/v1/models
```

---

### V1 Health Check
```bash
GET /api/v1/health
```
Extended health check including model availability.

```bash
curl http://localhost:8000/api/v1/health
```

---

## Interactive Documentation

FastAPI automatically generates interactive API docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 404 | Resource not found (pipeline not run yet) |
| 422 | Validation error (bad request body) |
| 500 | Internal server error |
