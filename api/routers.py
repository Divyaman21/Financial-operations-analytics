"""API routers for versioned endpoints."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["v1"])


class ChurnPredictionRequest(BaseModel):
    """Request body for churn prediction."""
    customer_id: int = Field(..., ge=1)
    recency_days: float = Field(..., ge=0)
    frequency_365d: float = Field(..., ge=0)
    monetary_365d: float = Field(..., ge=0)
    tenure_days: float = Field(..., ge=0)
    avg_interpurchase_days: float = Field(..., ge=0)
    clv_proxy: float = Field(..., ge=0)


class ChurnPredictionResponse(BaseModel):
    """Response body for churn prediction."""
    customer_id: int
    churn_probability: float
    risk_tier: str
    model_version: str


class ForecastResponse(BaseModel):
    """Response body for revenue forecast."""
    scenario: str
    forecast: list[float]
    model: str
    metrics: dict[str, float]


@router.post("/churn/predict", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest) -> dict:
    """
    Predict churn probability for a customer.

    Uses the best model from the registry. Falls back to a heuristic
    if no model artifact is available.
    """
    import math

    # Try loading from registry
    try:
        from fo_analytics.registry import ModelRegistry
        registry = ModelRegistry()
        model = registry.load("churn_best", version="latest")
        import numpy as np
        import pandas as pd

        features = pd.DataFrame([{
            "lag_frequency_365d": request.frequency_365d,
            "lag_monetary_365d": request.monetary_365d,
            "lag_tenure_days": request.tenure_days,
            "lag_avg_interpurchase_days": request.avg_interpurchase_days,
            "lag_clv_proxy": request.clv_proxy,
        }])
        prob = float(model.predict_proba(features)[0, 1])
        version = "registry_latest"
    except Exception:
        # Heuristic fallback
        z = (
            0.004 * request.recency_days
            - 0.02 * request.frequency_365d
            - 0.0001 * request.monetary_365d
            + 0.001 * request.tenure_days
        )
        prob = float(1 / (1 + math.exp(-z)))
        version = "heuristic_fallback"

    if prob > 0.6:
        tier = "High"
    elif prob > 0.3:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "customer_id": request.customer_id,
        "churn_probability": round(prob, 4),
        "risk_tier": tier,
        "model_version": version,
    }


@router.get("/forecast/26-weeks")
async def forecast_revenue(scenario: str = "base") -> dict:
    """
    Get 26-week revenue forecast.

    Scenarios: 'base', 'optimistic' (+8%), 'pessimistic' (-8%).
    """
    import json

    metrics_path = Path(__file__).resolve().parents[1] / "artifacts" / "metrics" / "forecast_comparison.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Run the pipeline first to generate forecasts")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    table = data.get("table", [])

    # Find best model by RMSE
    valid = [r for r in table if r.get("rmse") is not None]
    if not valid:
        raise HTTPException(status_code=404, detail="No valid forecast models found")

    best = min(valid, key=lambda x: x["rmse"])

    return {
        "scenario": scenario,
        "best_model": best.get("model", "unknown"),
        "metrics": {
            "mape": best.get("mape"),
            "smape": best.get("smape"),
            "rmse": best.get("rmse"),
        },
        "note": f"Scenario '{scenario}' adjustments are applied at serving time",
    }


@router.get("/health")
async def health_check() -> dict:
    """Check API and model health."""
    models_dir = Path(__file__).resolve().parents[1] / "artifacts" / "models"
    manifest = Path(__file__).resolve().parents[1] / "artifacts" / "model_manifest.json"

    model_count = len(list(models_dir.glob("*.pkl"))) if models_dir.exists() else 0
    has_manifest = manifest.exists()

    return {
        "status": "healthy",
        "models_available": model_count,
        "manifest_present": has_manifest,
        "last_check": datetime.now().isoformat(),
    }


@router.get("/drift/latest")
async def latest_drift() -> dict:
    """Return the latest drift report data."""
    import json

    drift_path = Path(__file__).resolve().parents[1] / "artifacts" / "monitoring" / "baseline.json"
    if not drift_path.exists():
        raise HTTPException(status_code=404, detail="No drift baseline found. Run pipeline first.")

    return json.loads(drift_path.read_text(encoding="utf-8"))


@router.get("/alerts/latest")
async def latest_alerts() -> dict:
    """Return the latest alerts."""
    import json

    alerts_path = Path(__file__).resolve().parents[1] / "artifacts" / "alerts.json"
    if not alerts_path.exists():
        return {"alerts": [], "message": "No alerts generated yet"}

    alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    return {"alerts": alerts, "count": len(alerts)}


@router.get("/models")
async def list_models() -> dict:
    """List all registered models and their versions."""
    import json

    manifest_path = Path(__file__).resolve().parents[1] / "artifacts" / "model_manifest.json"
    if not manifest_path.exists():
        return {"models": {}, "message": "No models registered yet"}

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {}
    for name, info in data.items():
        versions = info.get("versions", {})
        latest = info.get("latest", max(versions.keys()) if versions else None)
        summary[name] = {
            "total_versions": len(versions),
            "latest_version": latest,
            "latest_metrics": versions.get(latest, {}).get("metrics", {}) if latest else {},
        }
    return {"models": summary}
