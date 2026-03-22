"""Monte Carlo simulation on order-level revenue (sample or full)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_order_profit(
    orders: pd.DataFrame,
    n_draws: int = 5000,
    n_orders: int | None = 5000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    rev = orders["revenue"].astype(float).values
    if n_orders is not None and len(rev) > n_orders:
        rev = rng.choice(rev, size=n_orders, replace=False)
    # Assume cost ratio ~ lognormal around 0.65 of revenue (demo)
    cost_ratio = rng.lognormal(mean=-0.45, sigma=0.15, size=(n_draws, len(rev)))
    cost_ratio = np.clip(cost_ratio, 0.2, 0.95)
    profit_samples = (rev * (1 - cost_ratio)).sum(axis=1)
    return {
        "n_orders_in_sim": int(len(rev)),
        "n_draws": int(n_draws),
        "mean_total_profit": float(profit_samples.mean()),
        "p5": float(np.percentile(profit_samples, 5)),
        "p50": float(np.percentile(profit_samples, 50)),
        "p95": float(np.percentile(profit_samples, 95)),
        "std": float(profit_samples.std()),
    }
