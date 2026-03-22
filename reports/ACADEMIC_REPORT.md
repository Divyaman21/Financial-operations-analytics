# Financial Operations Analytics System

## Full academic-style report (IMRAD skeleton)

*Expand this skeleton to ~22 pages by pasting pipeline outputs from `artifacts/figures/`, `artifacts/tables/`, and `artifacts/metrics/`, and by adding citations.*

### Abstract

Summarize objectives (forecast, churn, profitability), data window, methods (HW, SARIMA, Prophet; classifiers + boosting; KM/Cox; RFM + K-Means; Monte Carlo), key metrics (holdout MAPE/sMAPE/RMSE; AUC-ROC/PR-AUC; R²; simulation intervals), and limitations.

### 1. Introduction

- Business context for financial operations analytics.  
- Research questions: multi-step revenue accuracy, churn discrimination, profit drivers, segment heterogeneity.  
- Contributions: reproducible pipeline, diagnostic rigor (AIC/BIC, Ljung–Box), stakeholder artifacts.

### 2. Data

- Sources: transactional orders, customer ids, timestamps, revenue.  
- Churn definition and censoring.  
- EDA: seasonality, concentration, missingness.  
- Ethics / privacy: pseudonymous ids, aggregation.

### 3. Methods

#### 3.1 Forecasting

- Holt–Winters formulation; SARIMAX grid + information criteria; Prophet holidays/changepoints.  
- Holdout protocol (26 weeks).  
- Residual diagnostics (Ljung–Box, ACF/QQ references to figures).

#### 3.2 Churn classification

- Feature horizon tied to `as_of_date`.  
- Models: logistic regression, RF, HGB, tuned XGB/LGBM with CV.  
- Metrics: ROC-AUC, PR-AUC, calibration discussion.  
- CLV-weighting scheme (log proxy).

#### 3.3 Survival analysis

- KM estimator; Cox PH; proportional hazards testing.

#### 3.4 Segmentation

- RFM rules; K-Means with silhouette vs interpretability; cross-tab vs RFM.

#### 3.5 Profitability

- OLS on log revenue; interpret coefficients cautiously.  
- Monte Carlo: cost ratio assumptions, distributional choices, convergence notes.

#### 3.6 BI & dashboard

- Star schema export; static HTML executive view; drill-down paths in Power BI/Tableau.

### 4. Results

- Forecast comparison table + plots (`forecast_holdout.png`, residual QQ).  
- Churn leaderboard + calibration.  
- KM curves; Cox table (`cox_summary.csv`).  
- Cluster × RFM heatmap.  
- Regression R²; MC p5/p50/p95.

### 5. Discussion

- Model choice reasoning; failure modes (sparse series, weak AUC).  
- Leakage audit checklist.  
- Non-goals (causal claims without design).

### 6. Conclusion

- Operational recommendations; monitoring plan; next work (nested CV, Bayesian forecasting, causal uplift).

### References

Add your sources (statsmodels, Prophet, lifelines, sklearn, etc.).

### Appendix A — Selected code

Point to `src/fo_analytics/` modules and `run_pipeline.py`.

### Appendix B — User testing

Summarize sessions from `docs/USER_TESTING.md`.
