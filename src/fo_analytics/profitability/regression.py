"""Profit driver OLS-style regression on customer-year aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def profit_driver_regression(customers: pd.DataFrame) -> dict:
    df = customers.dropna(
        subset=["monetary_365d", "frequency_365d", "recency_days", "tenure_days"]
    ).copy()
    y = np.log1p(df["monetary_365d"].clip(lower=0))
    X = df[["frequency_365d", "recency_days", "tenure_days"]].astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return {
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "summary_text": model.summary().as_text(),
    }
