"""Synthetic transactional data for end-to-end pipeline demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fo_analytics.config import RANDOM_SEED


def generate_orders(
    n_customers: int = 3000,
    start: str = "2022-01-01",
    end: str = "2024-03-01",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = (end_ts - start_ts).days

    rows: list[dict] = []
    categories = ["A", "B", "C"]

    for cid in range(1, n_customers + 1):
        # Heterogeneous activity: some frequent, some dormant
        base_rate = rng.lognormal(mean=-2.2, sigma=0.8)
        churn_propensity = rng.beta(2, 5)
        clv_scale = rng.lognormal(3.0, 0.5)

        t = 0
        while t < days:
            if rng.random() < churn_propensity * 0.002:
                t += rng.integers(60, 200)
                continue
            day_offset = start_ts + pd.Timedelta(days=int(t))
            if day_offset > end_ts:
                break
            noise = rng.normal(0, 8)
            revenue = max(5.0, clv_scale * rng.lognormal(0, 0.35) + noise)
            rows.append(
                {
                    "customer_id": cid,
                    "order_date": day_offset.normalize(),
                    "revenue": float(revenue),
                    "category": rng.choice(categories),
                }
            )
            gap = max(1, int(rng.exponential(1.0 / max(base_rate, 1e-6))))
            t += gap

    orders = pd.DataFrame(rows)
    orders = orders.sort_values(["order_date", "customer_id"]).reset_index(drop=True)
    return orders
