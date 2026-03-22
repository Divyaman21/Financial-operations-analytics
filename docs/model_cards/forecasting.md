# Model card — revenue forecasting

## Intent

Support **weekly revenue planning** with transparent comparison across seasonal baselines.

## Data

- **Input:** `orders_fact` aggregated to **week-start Monday** (`W-MON`) sums.  
- **Holdout:** last **26 weeks** withheld for error metrics.  
- **Leakage controls:** no post-holdout orders in training.

## Models

- Holt–Winters (`statsmodels`)  
- SARIMAX (`statsmodels`), **AIC/BIC** logged on training fit; grid over small `(p,d,q) × (P,D,Q,s)` space.  
- Prophet (`prophet`) with **US holidays** + optional custom holidays.

## Metrics

- **MAPE, sMAPE, RMSE** on holdout.  
- **Ljung–Box** p-value on in-sample residuals where applicable (SARIMA/HW).

## Limitations

- Sparse or highly intermittent series weaken Prophet and seasonal ARIMA.  
- AIC/BIC are **not directly comparable** across HW vs Prophet; prioritize **hold-out error** + residual diagnostics.

## Monitoring / retrain

- **Retrain** monthly or when MAPE drifts > agreed threshold vs rolling holdout.  
- Log `artifacts/metrics/forecast_comparison.json` per run.
