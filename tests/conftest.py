"""Shared test fixtures for the Financial Operations Analytics test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from fo_analytics.config import RANDOM_SEED
from fo_analytics.data.synthetic import generate_orders
from fo_analytics.evaluation.metrics import weekly_revenue_series
from fo_analytics.features.customers import build_customer_features
from fo_analytics.profitability import add_rfm_segments


@pytest.fixture(scope="session")
def orders() -> pd.DataFrame:
    """Generate a small reproducible orders dataset for all tests."""
    return generate_orders(n_customers=200, seed=RANDOM_SEED)


@pytest.fixture(scope="session")
def as_of(orders: pd.DataFrame) -> pd.Timestamp:
    """Observation date derived from the orders dataset."""
    return pd.to_datetime(orders["order_date"]).max().normalize()


@pytest.fixture(scope="session")
def weekly(orders: pd.DataFrame) -> pd.Series:
    """Weekly revenue series."""
    return weekly_revenue_series(orders)


@pytest.fixture(scope="session")
def customers(orders: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Customer features with RFM segments."""
    cust = build_customer_features(orders, as_of)
    cust = add_rfm_segments(cust)
    return cust
