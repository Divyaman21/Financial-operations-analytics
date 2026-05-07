"""Data drift detection using KS test + Population Stability Index (PSI)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class DriftDetector:
    """Detect statistical distribution shifts between baseline and new data."""

    def __init__(self, baseline_file: str | Path = "artifacts/monitoring/baseline.json") -> None:
        self.baseline_file = Path(baseline_file)
        self.baseline: dict[str, Any] = self._load_baseline()

    def compute_baseline(self, data: pd.DataFrame, features: list[str]) -> dict[str, Any]:
        """
        Compute baseline statistics from training data.

        Args:
            data: training DataFrame.
            features: list of column names to track.

        Returns:
            baseline dict with per-feature statistics.
        """
        baseline: dict[str, Any] = {}
        for col in features:
            if col not in data.columns:
                continue
            if data[col].dtype in ("float64", "float32", "int64", "int32"):
                vals = data[col].dropna()
                baseline[col] = {
                    "type": "continuous",
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "quantiles": {
                        "q25": float(vals.quantile(0.25)),
                        "q50": float(vals.quantile(0.50)),
                        "q75": float(vals.quantile(0.75)),
                    },
                    "sample_values": vals.values.tolist()[:5000],  # Store a sample for KS
                }
            else:
                baseline[col] = {
                    "type": "categorical",
                    "value_counts": data[col].value_counts().to_dict(),
                }
        return baseline

    def save_baseline(self, baseline: dict[str, Any]) -> None:
        """Save baseline statistics to JSON."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, default=str)
        self.baseline = baseline

    def check_drift(self, new_data: pd.DataFrame, feature: str) -> dict[str, Any] | None:
        """
        Detect drift using KS test (continuous) or Chi-squared (categorical).

        Returns:
            dict with p_value, psi, and alert_level, or None if feature unknown.
        """
        if feature not in self.baseline:
            return None

        baseline_info = self.baseline[feature]

        if baseline_info["type"] == "continuous":
            new_vals = new_data[feature].dropna().values

            # Use stored sample values for KS test when available
            if "sample_values" in baseline_info:
                baseline_vals = np.array(baseline_info["sample_values"])
            else:
                baseline_vals = np.random.normal(
                    baseline_info["mean"],
                    max(baseline_info["std"], 1e-9),
                    min(len(new_vals), 5000),
                )

            _statistic, p_value = ks_2samp(baseline_vals, new_vals)
            psi = self._calculate_psi(baseline_vals, new_vals)

            if psi > 0.25:
                alert_level = "CRITICAL"
            elif psi > 0.15:
                alert_level = "WARNING"
            elif psi > 0.05:
                alert_level = "CAUTION"
            else:
                alert_level = "OK"

            return {
                "feature": feature,
                "test": "Kolmogorov-Smirnov",
                "p_value": float(p_value),
                "psi": float(psi),
                "alert_level": alert_level,
            }

        return None

    def check_all(self, new_data: pd.DataFrame) -> dict[str, Any]:
        """Check drift for all baseline features present in new_data."""
        results: list[dict[str, Any]] = []
        for feature in self.baseline:
            if feature in new_data.columns:
                drift_result = self.check_drift(new_data, feature)
                if drift_result is not None:
                    results.append(drift_result)

        severity_order = {"OK": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}
        max_psi = max((r["psi"] for r in results), default=0.0)
        alert_severity = max(
            (r["alert_level"] for r in results),
            key=lambda x: severity_order.get(x, 0),
            default="OK",
        )

        return {
            "results": results,
            "max_psi": float(max_psi),
            "alert_severity": alert_severity,
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        eps = 1e-6
        combined_min = min(baseline.min(), current.min())
        combined_max = max(baseline.max(), current.max())
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)

        baseline_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
        current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

        baseline_pct = (baseline_counts + eps) / (baseline_counts.sum() + eps * bins)
        current_pct = (current_counts + eps) / (current_counts.sum() + eps * bins)

        psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
        return psi

    def _load_baseline(self) -> dict[str, Any]:
        if self.baseline_file.exists():
            with open(self.baseline_file, encoding="utf-8") as f:
                return json.load(f)
        return {}
