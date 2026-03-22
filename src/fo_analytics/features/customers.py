"""Customer-level features and labels with fixed observation dates (leakage-aware churn)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fo_analytics.config import CHURN_FEATURE_LAG_DAYS, INACTIVE_DAYS_CHURN


def _snapshot(orders: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """All behavioral columns as of ``as_of`` (inclusive)."""
    past = orders[orders["order_date"] <= as_of]
    first = past.groupby("customer_id")["order_date"].min().rename("first_purchase")
    last = past.groupby("customer_id")["order_date"].max().rename("last_purchase")
    freq_365 = (
        past[past["order_date"] > as_of - pd.Timedelta(days=365)]
        .groupby("customer_id")
        .size()
        .rename("frequency_365d")
    )
    monetary_365 = (
        past[past["order_date"] > as_of - pd.Timedelta(days=365)]
        .groupby("customer_id")["revenue"]
        .sum()
        .rename("monetary_365d")
    )
    monetary_all = past.groupby("customer_id")["revenue"].sum().rename("monetary_all")

    df = pd.concat([first, last, freq_365, monetary_365, monetary_all], axis=1)
    df["frequency_365d"] = df["frequency_365d"].fillna(0)
    df["monetary_365d"] = df["monetary_365d"].fillna(0.0)
    df["monetary_all"] = df["monetary_all"].fillna(0.0)

    df["recency_days"] = (as_of - df["last_purchase"]).dt.days
    df["tenure_days"] = (df["last_purchase"] - df["first_purchase"]).dt.days.clip(lower=0)

    sorted_p = past.sort_values(["customer_id", "order_date"])
    sorted_p = sorted_p.assign(
        _gap=sorted_p.groupby("customer_id")["order_date"].diff().dt.days
    )
    gap_mean = sorted_p.groupby("customer_id")["_gap"].mean().rename("avg_interpurchase_days")
    df = df.join(gap_mean, how="left")

    df["avg_interpurchase_days"] = df["avg_interpurchase_days"].fillna(df["tenure_days"].clip(upper=1))
    df["clv_proxy"] = df["monetary_365d"] * (df["frequency_365d"] + 1) / (df["recency_days"] + 1)
    return df


def build_customer_features(
    orders: pd.DataFrame,
    as_of: pd.Timestamp,
    inactive_days: int = INACTIVE_DAYS_CHURN,
    churn_feature_lag_days: int = CHURN_FEATURE_LAG_DAYS,
) -> pd.DataFrame:
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    snap_now = _snapshot(orders, as_of)
    snap_now["churned"] = (snap_now["recency_days"] > inactive_days).astype(int)

    lag_ts = as_of - pd.Timedelta(days=churn_feature_lag_days)
    snap_lag = _snapshot(orders, lag_ts)
    lag_cols = {
        "frequency_365d": "lag_frequency_365d",
        "monetary_365d": "lag_monetary_365d",
        "tenure_days": "lag_tenure_days",
        "avg_interpurchase_days": "lag_avg_interpurchase_days",
        "clv_proxy": "lag_clv_proxy",
    }
    snap_lag = snap_lag[list(lag_cols.keys())].rename(columns=lag_cols)

    df = snap_now.join(snap_lag, how="left")
    for c in lag_cols.values():
        df[c] = df[c].fillna(0.0)

    churn_date = df["last_purchase"] + pd.Timedelta(days=inactive_days)
    event_occurred = df["churned"] == 1
    duration = np.where(
        event_occurred,
        (churn_date - df["first_purchase"]).dt.days.clip(lower=1),
        (as_of - df["first_purchase"]).dt.days.clip(lower=1),
    )
    df["duration_days"] = duration.astype(int)
    df["event_observed"] = event_occurred.astype(int)

    return df.reset_index()
