"""Tests for churn classification models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fo_analytics.churn.models import (
    FEATURE_COLS,
    ChurnModelResult,
    refit_best_and_score,
    results_to_dataframe,
    train_churn_models,
)
from fo_analytics.config import RANDOM_SEED


class TestTrainChurnModels:
    """Validate the churn model training pipeline."""

    def test_returns_results_and_summary(self, customers):
        results, summary = train_churn_models(customers, seed=RANDOM_SEED)
        assert isinstance(results, list)
        assert isinstance(summary, dict)
        assert len(results) >= 3  # At least LR, RF, HGB

    def test_all_results_are_typed(self, customers):
        results, _ = train_churn_models(customers, seed=RANDOM_SEED)
        for r in results:
            assert isinstance(r, ChurnModelResult)

    def test_auc_roc_above_minimum(self, customers):
        results, _ = train_churn_models(customers, seed=RANDOM_SEED)
        for r in results:
            assert r.auc_roc >= 0.50, (
                f"{r.name} AUC-ROC={r.auc_roc:.3f} is below random chance"
            )

    def test_pr_auc_positive(self, customers):
        results, _ = train_churn_models(customers, seed=RANDOM_SEED)
        for r in results:
            assert r.pr_auc > 0.0, f"{r.name} PR-AUC must be positive"

    def test_probabilities_valid_range(self, customers):
        results, _ = train_churn_models(customers, seed=RANDOM_SEED)
        for r in results:
            assert np.all(r.prob_test >= 0.0) and np.all(r.prob_test <= 1.0), (
                f"{r.name} probabilities out of [0,1] range"
            )

    def test_summary_has_calibration(self, customers):
        _, summary = train_churn_models(customers, seed=RANDOM_SEED)
        assert "calibration_best_model" in summary
        assert "calibration_bins_mean_pred" in summary

    def test_summary_has_split_info(self, customers):
        _, summary = train_churn_models(customers, seed=RANDOM_SEED)
        assert "n_train" in summary
        assert "n_test" in summary
        assert "positive_rate" in summary
        assert 0.0 < summary["positive_rate"] < 1.0


class TestResultsToDataframe:
    """Validate conversion of results to DataFrame."""

    def test_sorted_by_auc(self, customers):
        results, _ = train_churn_models(customers, seed=RANDOM_SEED)
        df = results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert df["auc_roc"].is_monotonic_decreasing

    def test_required_columns(self, customers):
        results, _ = train_churn_models(customers, seed=RANDOM_SEED)
        df = results_to_dataframe(results)
        assert "model" in df.columns
        assert "auc_roc" in df.columns
        assert "pr_auc" in df.columns


class TestRefitBest:
    """Validate refit and scoring on full data."""

    def test_refit_returns_scores(self, customers):
        results, summary = train_churn_models(customers, seed=RANDOM_SEED)
        df = results_to_dataframe(results)
        best_name = str(df.iloc[0]["model"])
        best_params = summary.get(f"{best_name}_best_params")
        scores, _clf = refit_best_and_score(customers, best_name, best_params)
        assert isinstance(scores, pd.DataFrame)
        assert "customer_id" in scores.columns
        assert "churn_prob" in scores.columns

    def test_refit_scores_valid_range(self, customers):
        scores, _clf = refit_best_and_score(customers, "hist_gradient_boosting", None)
        assert (scores["churn_prob"] >= 0.0).all()
        assert (scores["churn_prob"] <= 1.0).all()
