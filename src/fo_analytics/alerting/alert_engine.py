"""Alert engine: rule-based alerts for churn spikes, margin declines, forecast errors, and drift."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class AlertEngine:
    """Generate and manage business-rule alerts."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or self._default_config()
        self.alerts: list[dict[str, Any]] = []

    def check_churn_spike(
        self,
        current_rate: float,
        historical_mean: float,
        historical_std: float,
    ) -> None:
        """Alert if churn_rate > mean + 3*sigma."""
        sigma_mult = self.config.get("churn_sigma_multiplier", 3.0)
        threshold = historical_mean + (sigma_mult * historical_std)
        if current_rate > threshold:
            self.alerts.append({
                "rule": "churn_spike",
                "severity": "WARNING",
                "message": (
                    f"Churn rate {current_rate:.1%} exceeds threshold "
                    f"{threshold:.1%} (mean + {sigma_mult}σ)"
                ),
                "value": current_rate,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
            })

    def check_margin_threshold(
        self,
        current_margin: float,
        min_margin: float | None = None,
    ) -> None:
        """Alert if margin < threshold."""
        min_margin = min_margin or self.config.get("min_margin", -0.01)
        if current_margin < min_margin:
            self.alerts.append({
                "rule": "margin_decline",
                "severity": "CRITICAL",
                "message": (
                    f"Net margin {current_margin:.2%} below minimum {min_margin:.2%}"
                ),
                "value": current_margin,
                "threshold": min_margin,
                "timestamp": datetime.now().isoformat(),
            })

    def check_forecast_error(
        self,
        mape: float,
        threshold: float | None = None,
    ) -> None:
        """Alert if forecast MAPE exceeds threshold."""
        threshold = threshold or self.config.get("max_mape", 0.05)
        if mape > threshold:
            self.alerts.append({
                "rule": "forecast_error",
                "severity": "CAUTION",
                "message": f"Forecast MAPE {mape:.2%} exceeds {threshold:.2%}",
                "value": mape,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
            })

    def check_data_drift(
        self,
        max_psi: float,
        threshold: float | None = None,
    ) -> None:
        """Alert if data drift PSI exceeds threshold."""
        threshold = threshold or self.config.get("max_psi", 0.15)
        if max_psi > threshold:
            self.alerts.append({
                "rule": "data_drift",
                "severity": "WARNING",
                "message": f"Data drift PSI {max_psi:.3f} exceeds {threshold:.3f}",
                "value": max_psi,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat(),
            })

    def check_model_staleness(
        self,
        last_trained_iso: str,
        max_age_days: int | None = None,
    ) -> None:
        """Alert if model has not been retrained within max_age_days."""
        max_age = max_age_days or self.config.get("max_model_age_days", 30)
        last = datetime.fromisoformat(last_trained_iso)
        age = (datetime.now() - last).days
        if age > max_age:
            self.alerts.append({
                "rule": "model_staleness",
                "severity": "CAUTION",
                "message": f"Model last trained {age} days ago (limit: {max_age})",
                "value": age,
                "threshold": max_age,
                "timestamp": datetime.now().isoformat(),
            })

    def generate_report(
        self,
        output_file: str | Path = "artifacts/alerts.html",
    ) -> None:
        """Generate an HTML alert report."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        severity_styles = {
            "CRITICAL": ("red", "#ffebee"),
            "WARNING": ("orange", "#fff3e0"),
            "CAUTION": ("#f9a825", "#fffde7"),
            "INFO": ("#1565c0", "#e3f2fd"),
        }

        alert_blocks = []
        for a in self.alerts:
            sev = a["severity"]
            border, bg = severity_styles.get(sev, ("#999", "#f5f5f5"))
            alert_blocks.append(
                f'<div class="alert" style="border-left-color:{border};background:{bg};">'
                f'<strong>[{sev}]</strong> {a["message"]}'
                f'<br><span class="note">Rule: {a["rule"]} | Value: {a["value"]} | '
                f'Threshold: {a["threshold"]} | {a["timestamp"]}</span></div>'
            )

        status = "🟢 All Clear" if not self.alerts else f"🔴 {len(self.alerts)} Alert(s)"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Alerts Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }}
        h1 {{ font-size: 1.6rem; }}
        .alert {{ margin: 10px 0; padding: 12px 16px; border-left: 5px solid; border-radius: 4px; }}
        .note {{ font-size: 0.8rem; color: #666; }}
        .status {{ font-size: 1.2rem; margin: 1rem 0; }}
    </style>
</head>
<body>
    <h1>🚨 Financial Analytics — Alerts Report</h1>
    <p class="note">Generated: {datetime.now().isoformat()}</p>
    <p class="status">{status}</p>
    {"".join(alert_blocks) if alert_blocks else '<p>No alerts triggered.</p>'}
</body>
</html>
"""
        output_file.write_text(html, encoding="utf-8")

    def save_json(self, output_file: str | Path = "artifacts/alerts.json") -> None:
        """Save alerts as JSON."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.alerts, f, indent=2, default=str)

    @staticmethod
    def _default_config() -> dict[str, Any]:
        return {
            "churn_sigma_multiplier": 3.0,
            "min_margin": -0.01,
            "max_mape": 0.05,
            "max_psi": 0.15,
            "max_model_age_days": 30,
        }
