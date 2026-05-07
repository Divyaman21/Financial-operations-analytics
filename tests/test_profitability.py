"""Tests for profitability modules: regression, Monte Carlo, cohort retention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fo_analytics.config import RANDOM_SEED
from fo_analytics.profitability.cohort import cohort_retention_table
from fo_analytics.profitability.monte_carlo import monte_carlo_order_profit
from fo_analytics.profitability.regression import profit_driver_regression


class TestProfitDriverRegression:
    """Validate OLS profit-driver regression."""

    def test_returns_dict(self, customers):
        result = profit_driver_regression(customers)
        assert isinstance(result, dict)

    def test_rsquared_range(self, customers):
        result = profit_driver_regression(customers)
        assert 0.0 <= result["rsquared"] <= 1.0

    def test_has_summary_text(self, customers):
        result = profit_driver_regression(customers)
        assert "summary_text" in result
        assert len(result["summary_text"]) > 50

    def test_params_present(self, customers):
        result = profit_driver_regression(customers)
        assert "params" in result
        assert "const" in result["params"]

    def test_pvalues_valid(self, customers):
        result = profit_driver_regression(customers)
        for pval in result["pvalues"].values():
            assert 0.0 <= pval <= 1.0

    def test_info_criteria_present(self, customers):
        result = profit_driver_regression(customers)
        assert "aic" in result
        assert "bic" in result


class TestMonteCarlo:
    """Validate Monte Carlo order profit simulation."""

    def test_returns_dict(self, orders):
        result = monte_carlo_order_profit(orders, n_draws=100, n_orders=100, seed=RANDOM_SEED)
        assert isinstance(result, dict)

    def test_required_keys(self, orders):
        result = monte_carlo_order_profit(orders, n_draws=100, n_orders=100, seed=RANDOM_SEED)
        expected_keys = {"n_orders_in_sim", "n_draws", "mean_total_profit", "p5", "p50", "p95", "std"}
        assert expected_keys.issubset(set(result.keys()))

    def test_percentile_ordering(self, orders):
        result = monte_carlo_order_profit(orders, n_draws=500, n_orders=200, seed=RANDOM_SEED)
        assert result["p5"] <= result["p50"] <= result["p95"]

    def test_mean_profit_positive(self, orders):
        result = monte_carlo_order_profit(orders, n_draws=500, n_orders=200, seed=RANDOM_SEED)
        assert result["mean_total_profit"] > 0, "Expected positive mean profit"

    def test_std_positive(self, orders):
        result = monte_carlo_order_profit(orders, n_draws=500, n_orders=200, seed=RANDOM_SEED)
        assert result["std"] > 0

    def test_reproducibility(self, orders):
        r1 = monte_carlo_order_profit(orders, n_draws=100, n_orders=100, seed=42)
        r2 = monte_carlo_order_profit(orders, n_draws=100, n_orders=100, seed=42)
        assert r1["mean_total_profit"] == r2["mean_total_profit"]


class TestCohortRetention:
    """Validate cohort retention table."""

    def test_returns_dataframe(self, orders, as_of):
        result = cohort_retention_table(orders, as_of)
        assert isinstance(result, pd.DataFrame)

    def test_retention_values_bounded(self, orders, as_of):
        result = cohort_retention_table(orders, as_of)
        assert (result.values >= 0).all()
        assert (result.values <= 1.0 + 1e-9).all()

    def test_first_period_retention_is_one(self, orders, as_of):
        result = cohort_retention_table(orders, as_of)
        if 0 in result.columns:
            # Period 0 retention should be 1.0 for all cohorts
            assert (result[0] >= 0.99).all()
