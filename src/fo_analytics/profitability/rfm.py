"""RFM scoring and segment labels."""

from __future__ import annotations

import pandas as pd


def add_rfm_segments(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()

    def quintile_1_to_5(series: pd.Series, invert: bool = False) -> pd.Series:
        pct = series.rank(pct=True, method="first")
        q = pd.cut(pct, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0001], labels=[1, 2, 3, 4, 5]).astype(int)
        if invert:
            q = 6 - q
        return q

    df["R_score"] = quintile_1_to_5(df["recency_days"], invert=True)
    df["F_score"] = quintile_1_to_5(df["frequency_365d"], invert=False)
    df["M_score"] = quintile_1_to_5(df["monetary_365d"], invert=False)
    df["rfm_score"] = df["R_score"].astype(str) + df["F_score"].astype(str) + df["M_score"].astype(str)

    def segment(row: pd.Series) -> str:
        r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])
        if r >= 4 and f >= 4 and m >= 4:
            return "champions"
        if r >= 3 and f >= 3:
            return "loyal"
        if r >= 4 and f <= 2:
            return "new_or_promising"
        if r <= 2 and f >= 3:
            return "at_risk"
        if r <= 2 and f <= 2:
            return "hibernating"
        return "other"

    df["rfm_segment"] = df.apply(segment, axis=1)
    return df
