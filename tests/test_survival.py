"""Tests for survival analysis (Kaplan-Meier and Cox PH)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fo_analytics.survival.analysis import SurvivalArtifacts, run_survival_analysis


class TestSurvivalAnalysis:
    """Validate Kaplan-Meier and Cox PH results."""

    def test_returns_artifacts(self, customers):
        result = run_survival_analysis(customers)
        assert isinstance(result, SurvivalArtifacts)

    def test_km_plot_generated(self, customers, tmp_path):
        fig_dir = tmp_path / "figures"
        result = run_survival_analysis(customers, figures_dir=fig_dir)
        assert result.km_plot_path is not None
        assert Path(result.km_plot_path).exists()

    def test_cox_summary_generated(self, customers, tmp_path):
        fig_dir = tmp_path / "figures"
        tbl_dir = tmp_path / "tables"
        result = run_survival_analysis(
            customers, figures_dir=fig_dir, tables_dir=tbl_dir
        )
        if result.cox_summary_csv is not None:
            assert Path(result.cox_summary_csv).exists()
            cox_df = pd.read_csv(result.cox_summary_csv)
            assert len(cox_df) > 0

    def test_ph_test_pvalue_valid(self, customers, tmp_path):
        fig_dir = tmp_path / "figures"
        tbl_dir = tmp_path / "tables"
        result = run_survival_analysis(
            customers, figures_dir=fig_dir, tables_dir=tbl_dir
        )
        if result.ph_test_p is not None:
            assert 0.0 <= result.ph_test_p <= 1.0

    def test_insufficient_rows_handled(self):
        small = pd.DataFrame({
            "customer_id": [1, 2],
            "duration_days": [10, 20],
            "event_observed": [1, 0],
            "rfm_segment": ["a", "b"],
            "recency_days": [5, 10],
            "frequency_365d": [1, 2],
            "monetary_365d": [100, 200],
            "tenure_days": [30, 60],
        })
        result = run_survival_analysis(small)
        assert result.notes == "insufficient_rows"

    def test_km_plot_by_segment(self, customers, tmp_path):
        fig_dir = tmp_path / "figures"
        run_survival_analysis(customers, figures_dir=fig_dir)
        seg_plot = fig_dir / "kaplan_meier_by_segment.png"
        # Should exist if there are multiple RFM segments
        if customers["rfm_segment"].nunique() > 1:
            assert seg_plot.exists()
