"""Optional FastAPI service for batch-style scoring (demo)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Financial Operations Analytics API", version="0.1.0")


@app.get("/")
def read_root() -> dict:
    return {
        "message": "Welcome to the Financial Operations Analytics API",
        "documentation": "/docs",
        "latest_metrics": "/artifacts/latest_metrics",
        "health": "/health",
    }


class ScoreRequest(BaseModel):
    customer_id: int = Field(..., ge=1)
    recency_days: float = Field(..., ge=0)
    frequency_365d: float = Field(..., ge=0)
    monetary_365d: float = Field(..., ge=0)
    tenure_days: float = Field(..., ge=0)
    avg_interpurchase_days: float = Field(..., ge=0)
    clv_proxy: float = Field(..., ge=0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score/churn_stub")
def score_churn_stub(row: ScoreRequest) -> dict:
    """Heuristic risk score — replace with loaded model artifact in production."""
    import math

    z = (
        0.004 * row.recency_days
        - 0.02 * row.frequency_365d
        - 0.0001 * row.monetary_365d
        + 0.001 * row.tenure_days
    )
    prob = float(1 / (1 + math.exp(-z)))
    return {"customer_id": row.customer_id, "churn_prob_stub": prob, "note": "Not production-calibrated"}


@app.get("/artifacts/latest_metrics")
def latest_metrics() -> dict:
    root = Path(__file__).resolve().parents[1] / "artifacts" / "metrics"
    if not root.exists():
        raise HTTPException(status_code=404, detail="Run run_pipeline.py first")
    out = {}
    for p in sorted(root.glob("*.json")):
        import json

        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out
