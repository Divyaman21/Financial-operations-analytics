"""Simple cohort retention from first-purchase month."""

from __future__ import annotations

import pandas as pd


def cohort_retention_table(orders: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    o = orders.copy()
    o["order_date"] = pd.to_datetime(o["order_date"])
    o = o[o["order_date"] <= as_of]
    first = o.groupby("customer_id")["order_date"].min().reset_index(name="first_order")
    o = o.merge(first, on="customer_id")
    o["cohort"] = o["first_order"].dt.to_period("M").dt.to_timestamp()
    o["period"] = o["order_date"].dt.to_period("M").dt.to_timestamp()
    o["period_index"] = (o["period"].dt.year - o["cohort"].dt.year) * 12 + (
        o["period"].dt.month - o["cohort"].dt.month
    )
    cohort_sizes = o.groupby("cohort")["customer_id"].nunique()
    active = o.groupby(["cohort", "period_index"])["customer_id"].nunique().unstack(fill_value=0)
    ret = active.divide(cohort_sizes, axis=0)
    return ret.iloc[:12, :13].fillna(0)
