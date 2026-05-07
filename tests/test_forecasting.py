"""Tests for revenue forecasting models (Holt-Winters, SARIMA, Prophet)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fo_analytics.config import HOLDOUT_WEEKS
from fo_analytics.evaluation.metrics import mape, rmse, smape, weekly_revenue_series
from fo_analytics.forecasting.models import (
    ForecastResult,
    forecast_comparison,
    fit_predict_sarima,
    ljung_box_summary,
    split_series,
)


class TestSplitSeries:
    """Validate train/test splitting."""

    def test_split_lengths(self, weekly):
        holdout = 10
        train, test = split_series(weekly, holdout)
        assert len(test) == holdout
        assert len(train) + len(test) == len(weekly.dropna())

    def test_split_no_overlap(self, weekly):
        train, test = split_series(weekly, 10)
        assert train.index.max() < test.index.min()

    def test_split_raises_on_short_series(self):
        short = pd.Series(range(5), index=pd.date_range("2023-01-01", periods=5, freq="W-MON"))
        with pytest.raises(ValueError, match="too short"):
            split_series(short, 4)


class TestSARIMA:
    """Validate SARIMA fitting and prediction."""

    def test_sarima_output_length(self, weekly):
        train, test = split_series(weekly, HOLDOUT_WEEKS)
        pred, _res, _extras = fit_predict_sarima(train, len(test))
        assert len(pred) == len(test)

    def test_sarima_no_nans(self, weekly):
        train, test = split_series(weekly, HOLDOUT_WEEKS)
        pred, _res, _extras = fit_predict_sarima(train, len(test))
        assert not np.any(np.isnan(pred)), "SARIMA predictions should not contain NaN"

    def test_sarima_extras_keys(self, weekly):
        train, test = split_series(weekly, HOLDOUT_WEEKS)
        _pred, _res, extras = fit_predict_sarima(train, len(test))
        assert "order" in extras
        assert "seasonal_order" in extras
        assert "info_criteria" in extras


class TestLjungBox:
    """Validate Ljung-Box diagnostic."""

    def test_ljung_box_on_noise(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1, 100)
        result = ljung_box_summary(noise, lags=10)
        assert "ljung_box_p_at_lag" in result
        # White noise should pass (p > 0.05)
        assert result["pass_0_05"] is True

    def test_ljung_box_short_series(self):
        result = ljung_box_summary(np.array([1, 2, 3]), lags=10)
        assert result["ljung_box_min_p"] is None


class TestForecastComparison:
    """Validate the full forecast comparison workflow."""

    def test_returns_results_and_dataframe(self, weekly):
        results, comp = forecast_comparison(weekly, HOLDOUT_WEEKS)
        assert isinstance(results, list)
        assert isinstance(comp, pd.DataFrame)
        assert len(results) >= 2  # At least HW + SARIMA

    def test_comparison_table_columns(self, weekly):
        _results, comp = forecast_comparison(weekly, HOLDOUT_WEEKS)
        assert "model" in comp.columns
        assert "mape" in comp.columns
        assert "rmse" in comp.columns

    def test_each_result_has_metrics(self, weekly):
        results, _comp = forecast_comparison(weekly, HOLDOUT_WEEKS)
        for r in results:
            assert isinstance(r, ForecastResult)
            assert "mape" in r.metrics or "error" in r.extras

    def test_mape_reasonable(self, weekly):
        results, _comp = forecast_comparison(weekly, HOLDOUT_WEEKS)
        for r in results:
            if not np.any(np.isnan(r.y_pred)):
                m = r.metrics.get("mape", float("inf"))
                # MAPE should be < 100% for any reasonable model
                assert m < 1.0, f"{r.name} MAPE={m:.3f} is unreasonable"
