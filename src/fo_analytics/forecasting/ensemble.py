"""Forecast ensemble: combine HW, SARIMA, Prophet via weighted aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression


class ForecastEnsemble:
    """Combine multiple forecast models via data-driven weights."""

    def __init__(self) -> None:
        self.weights: dict[str, float] | None = None
        self.method: str = ""

    def fit_weights(
        self,
        y_true: np.ndarray,
        forecasts_dict: dict[str, np.ndarray],
        method: str = "inverse_variance",
    ) -> dict[str, float]:
        """
        Compute ensemble weights from holdout performance.

        Args:
            y_true: actual holdout values.
            forecasts_dict: {model_name: predicted_values}.
            method: 'equal', 'inverse_variance', or 'stacking'.

        Returns:
            dict of model_name → weight.
        """
        y_true = np.asarray(y_true, dtype=float)
        self.method = method

        # Filter out models with NaN predictions
        valid = {
            name: np.asarray(fc, dtype=float)
            for name, fc in forecasts_dict.items()
            if not np.any(np.isnan(fc)) and len(fc) == len(y_true)
        }

        if not valid:
            self.weights = {}
            return self.weights

        if method == "equal":
            n = len(valid)
            self.weights = {name: 1.0 / n for name in valid}

        elif method == "inverse_variance":
            rmses = {
                name: float(np.sqrt(np.mean((y_true - fc) ** 2)))
                for name, fc in valid.items()
            }
            # Avoid division by zero
            inv = {name: 1.0 / max(r, 1e-9) ** 2 for name, r in rmses.items()}
            total = sum(inv.values())
            self.weights = {name: w / total for name, w in inv.items()}

        elif method == "stacking":
            names = sorted(valid.keys())
            X_meta = np.column_stack([valid[n] for n in names])
            meta = LinearRegression()
            meta.fit(X_meta, y_true)
            raw_weights = meta.coef_
            # Normalise to sum to 1 (allow negative for correction)
            total = sum(abs(w) for w in raw_weights)
            self.weights = {
                name: float(coef / total) if total > 0 else 1.0 / len(names)
                for name, coef in zip(names, raw_weights)
            }
        else:
            raise ValueError(f"Unknown method: {method}")

        return self.weights

    def predict(
        self,
        forecasts_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Generate weighted ensemble forecast.

        Args:
            forecasts_dict: {model_name: forecast_array}.

        Returns:
            Weighted ensemble forecast array.
        """
        if self.weights is None:
            # Default to equal weights
            n = len(forecasts_dict)
            self.weights = {name: 1.0 / n for name in forecasts_dict}

        # Determine output length from the first valid forecast
        lengths = [len(fc) for fc in forecasts_dict.values()]
        h = lengths[0] if lengths else 0

        ensemble = np.zeros(h)
        for name, forecast in forecasts_dict.items():
            weight = self.weights.get(name, 0.0)
            fc = np.asarray(forecast, dtype=float)
            if len(fc) == h and not np.any(np.isnan(fc)):
                ensemble += weight * fc

        return ensemble

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of the ensemble configuration."""
        return {
            "method": self.method,
            "weights": self.weights or {},
        }
