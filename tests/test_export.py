"""Tests for BI export (CSV, Parquet) and metrics JSON output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fo_analytics.export.bi import write_bi_tables
from fo_analytics.io.metrics_log import append_metrics_log, write_json


class TestWriteBITables:
    """Validate CSV and Parquet BI exports."""

    @pytest.fixture
    def sample_tables(self):
        customers_dim = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "first_purchase": ["2022-01-01", "2022-06-15", "2023-01-10"],
            "rfm_segment": ["champions", "loyal", "at_risk"],
            "kmeans_cluster": [0, 1, 2],
        })
        orders_fact = pd.DataFrame({
            "customer_id": [1, 1, 2, 3],
            "order_date": ["2022-01-01", "2022-03-15", "2022-07-01", "2023-02-01"],
            "revenue": [100.0, 200.0, 150.0, 80.0],
        })
        model_scores = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "churn_prob": [0.1, 0.5, 0.9],
        })
        return customers_dim, orders_fact, model_scores

    def test_csv_files_created(self, sample_tables, tmp_path):
        c, o, m = sample_tables
        write_bi_tables(c, o, m, tmp_path)
        assert (tmp_path / "customers_dim.csv").exists()
        assert (tmp_path / "orders_fact.csv").exists()
        assert (tmp_path / "model_scores_fact.csv").exists()

    def test_parquet_files_created(self, sample_tables, tmp_path):
        c, o, m = sample_tables
        write_bi_tables(c, o, m, tmp_path)
        assert (tmp_path / "customers_dim.parquet").exists()
        assert (tmp_path / "orders_fact.parquet").exists()
        assert (tmp_path / "model_scores_fact.parquet").exists()

    def test_csv_roundtrip(self, sample_tables, tmp_path):
        c, o, m = sample_tables
        write_bi_tables(c, o, m, tmp_path)
        loaded = pd.read_csv(tmp_path / "customers_dim.csv")
        assert len(loaded) == len(c)
        assert set(loaded.columns) == set(c.columns)

    def test_parquet_roundtrip(self, sample_tables, tmp_path):
        c, o, m = sample_tables
        write_bi_tables(c, o, m, tmp_path)
        loaded = pd.read_parquet(tmp_path / "customers_dim.parquet")
        assert len(loaded) == len(c)


class TestMetricsLog:
    """Validate JSON metrics logging."""

    def test_write_json(self, tmp_path):
        path = tmp_path / "test.json"
        payload = {"accuracy": 0.95, "model": "xgboost"}
        write_json(path, payload)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["accuracy"] == 0.95

    def test_append_metrics_creates_dir(self, tmp_path):
        append_metrics_log(tmp_path, "test_metric", {"value": 42})
        assert (tmp_path / "metrics" / "test_metric.json").exists()

    def test_append_metrics_content(self, tmp_path):
        append_metrics_log(tmp_path, "forecast", {"mape": 0.03, "rmse": 500})
        data = json.loads((tmp_path / "metrics" / "forecast.json").read_text())
        assert data["mape"] == 0.03
        assert data["rmse"] == 500
