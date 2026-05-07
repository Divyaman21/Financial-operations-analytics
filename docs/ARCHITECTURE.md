# System Architecture

## Overview

The Financial Operations Analytics System is a modular, end-to-end Python application for business intelligence. It processes transactional order data through multiple analytical pipelines and produces actionable insights via dashboards, BI exports, and a REST API.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Synthetic    │    │  Feature          │    │  Weekly Revenue  │  │
│  │  Orders       │───▶│  Engineering      │───▶│  Series          │  │
│  │  Generator    │    │  (customers.py)   │    │  (metrics.py)    │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
└─────────────────────┬───────────────────────────────┬───────────────┘
                      │                               │
         ┌────────────▼────────────┐     ┌────────────▼────────────┐
         │     MODEL LAYER         │     │   FORECASTING LAYER     │
         │                         │     │                         │
         │  ┌─────────────────┐   │     │  ┌─────────────────┐   │
         │  │ Churn Models    │   │     │  │ Holt-Winters    │   │
         │  │ (5 classifiers) │   │     │  │ SARIMA          │   │
         │  ├─────────────────┤   │     │  │ Prophet         │   │
         │  │ Survival (KM,   │   │     │  │ Ensemble        │   │
         │  │ Cox PH)         │   │     │  └─────────────────┘   │
         │  ├─────────────────┤   │     └────────────┬────────────┘
         │  │ RFM + K-Means   │   │                  │
         │  ├─────────────────┤   │                  │
         │  │ Profit OLS      │   │                  │
         │  │ Monte Carlo     │   │                  │
         │  └─────────────────┘   │                  │
         └────────────┬───────────┘                  │
                      │                               │
         ┌────────────▼───────────────────────────────▼───────────────┐
         │                    MLOPS LAYER                              │
         │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
         │  │ Model        │  │ Drift        │  │ Alert            │ │
         │  │ Registry     │  │ Monitoring   │  │ Engine           │ │
         │  └──────────────┘  └──────────────┘  └──────────────────┘ │
         │  ┌──────────────────────────────────────────────────────┐  │
         │  │ SHAP Explainability                                  │  │
         │  └──────────────────────────────────────────────────────┘  │
         └────────────────────────────┬───────────────────────────────┘
                                      │
         ┌────────────────────────────▼───────────────────────────────┐
         │                    OUTPUT LAYER                             │
         │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
         │  │ HTML          │  │ BI Export    │  │ FastAPI          │ │
         │  │ Dashboard     │  │ (CSV/Parquet)│  │ REST API         │ │
         │  └──────────────┘  └──────────────┘  └──────────────────┘ │
         └────────────────────────────────────────────────────────────┘
```

## Module Descriptions

| Module | Path | Purpose |
|--------|------|---------|
| **Data** | `src/fo_analytics/data/` | Synthetic order generation with realistic distributions |
| **Features** | `src/fo_analytics/features/` | Customer-level feature engineering with lag-based churn labels |
| **Forecasting** | `src/fo_analytics/forecasting/` | Time-series models + ensemble aggregation |
| **Churn** | `src/fo_analytics/churn/` | 5 classifiers with CV tuning and calibration |
| **Survival** | `src/fo_analytics/survival/` | Kaplan-Meier and Cox PH analysis |
| **Segmentation** | `src/fo_analytics/segmentation/` | K-Means clustering on RFM features |
| **Profitability** | `src/fo_analytics/profitability/` | RFM, OLS regression, Monte Carlo, cohort retention |
| **Registry** | `src/fo_analytics/registry/` | Model serialization and versioned tracking |
| **Monitoring** | `src/fo_analytics/monitoring/` | Data drift detection (KS test + PSI) |
| **Alerting** | `src/fo_analytics/alerting/` | Rule-based business alerts |
| **Explainability** | `src/fo_analytics/explainability/` | SHAP-based model explanations |
| **Export** | `src/fo_analytics/export/` | Star-schema BI tables (CSV + Parquet) |
| **Dashboard** | `src/fo_analytics/dashboard/` | Static HTML executive dashboard |
| **IO** | `src/fo_analytics/io/` | JSON metrics logging |
| **API** | `api/` | FastAPI REST endpoints |

## Data Flow

1. **Ingest** → Generate or load transactional order data
2. **Transform** → Build customer features with time-lagged churn labels
3. **Model** → Train forecasting, churn, survival, and segmentation models
4. **Evaluate** → Compute holdout metrics, residual diagnostics
5. **Register** → Serialize models with versioned metadata
6. **Monitor** → Compute drift baselines and check for distribution shifts
7. **Alert** → Evaluate business rules and generate alerts
8. **Export** → Write BI-ready tables and HTML dashboard
9. **Serve** → Expose predictions and metrics via REST API

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Core | Python 3.9+, pandas, NumPy |
| ML | scikit-learn, XGBoost, LightGBM |
| Statistics | statsmodels, lifelines, Prophet |
| Explainability | SHAP |
| Visualisation | matplotlib |
| API | FastAPI, uvicorn |
| Container | Docker, docker-compose |
| CI/CD | GitHub Actions, pytest, codecov |
