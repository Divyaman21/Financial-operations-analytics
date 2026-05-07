"""Tests for RFM segmentation and K-Means clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fo_analytics.config import RANDOM_SEED
from fo_analytics.profitability.rfm import add_rfm_segments
from fo_analytics.segmentation.clustering import (
    attach_clusters,
    cluster_rfm_crosstab,
    kmeans_segmentation,
)


class TestRFMSegments:
    """Validate RFM scoring and segment assignment."""

    def test_rfm_columns_added(self, customers):
        assert "R_score" in customers.columns
        assert "F_score" in customers.columns
        assert "M_score" in customers.columns
        assert "rfm_segment" in customers.columns

    def test_rfm_scores_range(self, customers):
        for col in ["R_score", "F_score", "M_score"]:
            assert customers[col].min() >= 1
            assert customers[col].max() <= 5

    def test_rfm_segment_values(self, customers):
        valid = {"champions", "loyal", "new_or_promising", "at_risk", "hibernating", "other"}
        assert set(customers["rfm_segment"].unique()).issubset(valid)

    def test_all_customers_segmented(self, customers):
        assert customers["rfm_segment"].notna().all()

    def test_rfm_score_string_format(self, customers):
        # Each rfm_score should be 3 digits like "555" or "123"
        for score in customers["rfm_score"].dropna():
            assert len(str(score)) == 3
            assert str(score).isdigit()


class TestKMeans:
    """Validate K-Means clustering."""

    def test_kmeans_returns_series_and_meta(self, customers):
        feature_cols = ["recency_days", "frequency_365d", "monetary_365d", "tenure_days"]
        series, meta = kmeans_segmentation(customers, feature_cols, seed=RANDOM_SEED)
        assert isinstance(series, pd.Series)
        assert isinstance(meta, dict)

    def test_best_k_in_range(self, customers):
        feature_cols = ["recency_days", "frequency_365d", "monetary_365d", "tenure_days"]
        _, meta = kmeans_segmentation(customers, feature_cols, k_min=3, k_max=8, seed=RANDOM_SEED)
        assert 3 <= meta["best_k"] <= 8

    def test_silhouette_positive(self, customers):
        feature_cols = ["recency_days", "frequency_365d", "monetary_365d", "tenure_days"]
        _, meta = kmeans_segmentation(customers, feature_cols, seed=RANDOM_SEED)
        assert meta["silhouette_chosen"] > 0.0

    def test_attach_clusters(self, customers):
        feature_cols = ["recency_days", "frequency_365d", "monetary_365d", "tenure_days"]
        series, _ = kmeans_segmentation(customers, feature_cols, seed=RANDOM_SEED)
        result = attach_clusters(customers, series)
        assert "kmeans_cluster" in result.columns
        assert result["kmeans_cluster"].notna().all()

    def test_crosstab_shape(self, customers):
        feature_cols = ["recency_days", "frequency_365d", "monetary_365d", "tenure_days"]
        series, _ = kmeans_segmentation(customers, feature_cols, seed=RANDOM_SEED)
        cust = attach_clusters(customers, series)
        ct = cluster_rfm_crosstab(cust)
        assert isinstance(ct, pd.DataFrame)
        assert len(ct) > 0
