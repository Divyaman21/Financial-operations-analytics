# Model card — time-to-churn (survival)

## Intent

Describe **time-to-churn** from first purchase with censoring for active customers.

## Data

- **Duration:** days from first purchase to churn boundary or `as_of_date`.  
- **Event:** churn per inactive rule.  
- **Covariates (Cox):** recency, frequency, monetary, tenure (complete cases only).

## Methods

- **Kaplan–Meier** overall and by **RFM segment**.  
- **Cox PH** with light **L2 penalizer** for stability.  
- **Proportional hazards** global test p-value stored when available.

## Limitations

- Retail purchase gaps ≠ subscription churn; definition must match business.  
- Cox assumes PH; violations imply time-varying coefficients or alternative models.

## Monitoring

- Refresh KM/Cox on rolling windows; compare median survival by segment quarter-over-quarter.
