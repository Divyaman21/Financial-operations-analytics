# Model card — churn risk

## Intent

Prioritize retention outreach using interpretable features at a fixed **as-of** date.

## Data

- **Labels:** inactive **> 90 days** since last purchase at `as_of_date`.  
- **Features:** lagged snapshot (default **60 days** before `as_of_date`): trailing frequency/monetary, tenure, average inter-purchase gap, CLV proxy. Label uses full `as_of_date` inactivity rule.  
- **Split:** stratified **75/25** holdout; optional time-based split documented in metrics JSON.

## Models

- Logistic regression (scaled)  
- Random forest  
- HistGradientBoosting  
- **XGBoost** / **LightGBM** with **5-fold stratified GridSearchCV** (small grids; expand for production).

## Metrics

- **AUC-ROC**, **PR-AUC** (primary under imbalance).  
- **Calibration bins** for the best holdout model (see `artifacts/metrics/churn.json`).

## Limitations

- Refit scores exported to BI are **not** a substitute for out-of-time validation.  
- CLV proxy is a **heuristic** weight, not contractual CLV.

## Monitoring / retrain

- Track **PR-AUC** and **population stability** of `churn_prob` monthly.  
- Retrain when feature distributions shift or campaign mix changes materially.
