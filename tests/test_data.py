"""Tests for synthetic data generation and data pipeline integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fo_analytics.config import RANDOM_SEED
from fo_analytics.data.synthetic import generate_orders


class TestGenerateOrders:
    """Validate the synthetic order generator."""

    def test_returns_dataframe(self):
        df = generate_orders(n_customers=50, seed=RANDOM_SEED)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = generate_orders(n_customers=50, seed=RANDOM_SEED)
        expected = {"customer_id", "order_date", "revenue", "category"}
        assert expected.issubset(set(df.columns))

    def test_no_null_values(self):
        df = generate_orders(n_customers=50, seed=RANDOM_SEED)
        assert df.notna().all().all(), "Synthetic data should have no NaNs"

    def test_customer_id_range(self):
        n = 100
        df = generate_orders(n_customers=n, seed=RANDOM_SEED)
        assert df["customer_id"].min() >= 1
        assert df["customer_id"].max() <= n

    def test_revenue_positive(self):
        df = generate_orders(n_customers=50, seed=RANDOM_SEED)
        assert (df["revenue"] > 0).all(), "All revenue values must be positive"

    def test_date_range(self):
        start, end = "2022-01-01", "2024-03-01"
        df = generate_orders(n_customers=50, start=start, end=end, seed=RANDOM_SEED)
        dates = pd.to_datetime(df["order_date"])
        assert dates.min() >= pd.Timestamp(start)
        assert dates.max() <= pd.Timestamp(end)

    def test_categories_valid(self):
        df = generate_orders(n_customers=50, seed=RANDOM_SEED)
        assert set(df["category"].unique()).issubset({"A", "B", "C"})

    def test_reproducibility(self):
        df1 = generate_orders(n_customers=50, seed=42)
        df2 = generate_orders(n_customers=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_sorted_output(self):
        df = generate_orders(n_customers=50, seed=RANDOM_SEED)
        dates = pd.to_datetime(df["order_date"])
        assert dates.is_monotonic_increasing or (dates.diff().dropna() >= pd.Timedelta(0)).all()

    def test_minimum_row_count(self):
        df = generate_orders(n_customers=100, seed=RANDOM_SEED)
        # With 100 customers over 2+ years, we expect substantial rows
        assert len(df) > 100, f"Expected > 100 rows, got {len(df)}"
