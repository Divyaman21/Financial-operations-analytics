"""Integration test: run the full pipeline end-to-end and verify outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fo_analytics.config import RANDOM_SEED
from fo_analytics.data.synthetic import generate_orders
from fo_analytics.pipeline import run_all


class TestEndToEndPipeline:
    """Run the entire pipeline and validate artifact structure."""

    @pytest.fixture(scope="class")
    def pipeline_output(self, tmp_path_factory):
        """Run the full pipeline once into a temporary directory."""
        root = tmp_path_factory.mktemp("pipeline_artifacts")
        orders = generate_orders(n_customers=150, seed=RANDOM_SEED)
        run_all(orders=orders, artifacts_root=root)
        return root

    def test_artifacts_root_exists(self, pipeline_output):
        assert pipeline_output.exists()

    def test_figures_directory(self, pipeline_output):
        fig_dir = pipeline_output / "figures"
        assert fig_dir.exists()
        png_files = list(fig_dir.glob("*.png"))
        assert len(png_files) >= 3, f"Expected >= 3 figures, found {len(png_files)}"

    def test_forecast_holdout_plot(self, pipeline_output):
        assert (pipeline_output / "figures" / "forecast_holdout.png").exists()

    def test_tables_directory(self, pipeline_output):
        tbl_dir = pipeline_output / "tables"
        assert tbl_dir.exists()
        csv_files = list(tbl_dir.glob("*.csv"))
        assert len(csv_files) >= 3

    def test_forecast_comparison_csv(self, pipeline_output):
        path = pipeline_output / "tables" / "forecast_comparison.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert "model" in df.columns
        assert len(df) >= 2

    def test_churn_models_csv(self, pipeline_output):
        path = pipeline_output / "tables" / "churn_models.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert "auc_roc" in df.columns

    def test_cohort_retention_csv(self, pipeline_output):
        assert (pipeline_output / "tables" / "cohort_retention.csv").exists()

    def test_bi_export_directory(self, pipeline_output):
        bi_dir = pipeline_output / "bi_export"
        assert bi_dir.exists()
        assert (bi_dir / "customers_dim.csv").exists()
        assert (bi_dir / "orders_fact.csv").exists()
        assert (bi_dir / "model_scores_fact.csv").exists()
        assert (bi_dir / "customers_dim.parquet").exists()

    def test_dashboard_generated(self, pipeline_output):
        dash = pipeline_output / "dashboard" / "index.html"
        assert dash.exists()
        html = dash.read_text(encoding="utf-8")
        assert "Financial Operations Analytics" in html

    def test_metrics_json_files(self, pipeline_output):
        metrics_dir = pipeline_output / "metrics"
        assert metrics_dir.exists()
        json_files = list(metrics_dir.glob("*.json"))
        assert len(json_files) >= 4, f"Expected >= 4 metrics files, found {len(json_files)}"
