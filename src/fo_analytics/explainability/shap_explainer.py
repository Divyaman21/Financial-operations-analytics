"""SHAP-based model explainability for tree-based churn classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/pipeline use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SHAPExplainer:
    """Generate SHAP explanations for tree-based models."""

    def __init__(self, model: Any, X_train: pd.DataFrame) -> None:
        """
        Initialise the SHAP explainer.

        Args:
            model: a fitted tree-based model (XGBoost, LightGBM, RF, HGB, etc.).
            X_train: training features used to build the background distribution.
        """
        import shap

        self.model = model
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            # Fallback for models not supported by TreeExplainer
            self.explainer = shap.Explainer(model, X_train)

        self._train_shap_values = self.explainer.shap_values(X_train)
        # For binary classifiers, shap_values may return a list; take positive class
        if isinstance(self._train_shap_values, list) and len(self._train_shap_values) == 2:
            self._train_shap_values = self._train_shap_values[1]

    def summary_plot(
        self,
        X: pd.DataFrame,
        output_file: str | Path | None = None,
    ) -> None:
        """Generate and optionally save a SHAP summary bar plot."""
        import shap

        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (mean |SHAP|)")
        plt.tight_layout()

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

    def beeswarm_plot(
        self,
        X: pd.DataFrame,
        output_file: str | Path | None = None,
    ) -> None:
        """Generate and optionally save a SHAP beeswarm plot."""
        import shap

        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X, show=False)
        plt.title("SHAP Beeswarm — feature impact on churn prediction")
        plt.tight_layout()

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

    def explain_instance(self, X_instance: pd.DataFrame) -> dict[str, Any]:
        """
        Return top driving features for a single prediction.

        Args:
            X_instance: a single-row DataFrame.

        Returns:
            dict with 'top_features' and 'contributions'.
        """
        shap_vals = self.explainer.shap_values(X_instance)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

        vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

        feature_names = (
            list(X_instance.columns) if hasattr(X_instance, "columns") else self.feature_names
        )
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(vals))]

        pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)

        return {
            "top_features": [p[0] for p in pairs[:5]],
            "contributions": [float(p[1]) for p in pairs[:5]],
        }

    def global_importance(self) -> dict[str, float]:
        """Return mean absolute SHAP value per feature (global importance)."""
        mean_abs = np.abs(self._train_shap_values).mean(axis=0)
        names = self.feature_names or [f"feature_{i}" for i in range(len(mean_abs))]
        return dict(sorted(zip(names, mean_abs.tolist()), key=lambda x: x[1], reverse=True))
