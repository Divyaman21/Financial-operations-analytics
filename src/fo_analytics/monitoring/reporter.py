"""Generate HTML drift report from drift detection results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def generate_drift_report(
    drift_results: dict[str, Any],
    output_file: str | Path = "artifacts/monitoring/drift_report.html",
) -> None:
    """Generate an HTML report summarising data drift analysis."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = drift_results.get("results", [])
    alert_severity = drift_results.get("alert_severity", "OK")
    max_psi = drift_results.get("max_psi", 0.0)
    timestamp = drift_results.get("timestamp", datetime.now().isoformat())

    alert_class = "ok"
    if alert_severity == "CRITICAL":
        alert_class = "critical"
    elif alert_severity == "WARNING":
        alert_class = "warning"
    elif alert_severity == "CAUTION":
        alert_class = "caution"

    rows_html = ""
    for r in results:
        level = r.get("alert_level", "OK")
        row_class = level.lower()
        rows_html += f"""
            <tr class="{row_class}">
                <td>{r.get("feature", "")}</td>
                <td>{r.get("test", "")}</td>
                <td>{r.get("p_value", 0):.4f}</td>
                <td>{r.get("psi", 0):.4f}</td>
                <td><strong>{level}</strong></td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Data Drift Report</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }}
        h1 {{ font-size: 1.6rem; }}
        .summary {{ margin: 1rem 0; padding: 1rem; border-radius: 8px; }}
        .ok {{ background: #e8f5e9; }}
        .caution {{ background: #fff8e1; }}
        .warning {{ background: #fff3e0; }}
        .critical {{ background: #ffebee; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 12px; text-align: left; }}
        th {{ background-color: #263238; color: white; }}
        tr:nth-child(even) {{ background-color: #f5f5f5; }}
        .note {{ font-size: 0.85rem; color: #555; }}
    </style>
</head>
<body>
    <h1>📊 Data Drift Analysis Report</h1>
    <p class="note">Generated: {timestamp}</p>

    <div class="summary {alert_class}">
        <strong>Overall Alert Level:</strong> {alert_severity} &nbsp;|&nbsp;
        <strong>Max PSI:</strong> {max_psi:.4f} &nbsp;|&nbsp;
        <strong>Features Checked:</strong> {len(results)}
    </div>

    <table>
        <tr>
            <th>Feature</th>
            <th>Test</th>
            <th>P-Value</th>
            <th>PSI</th>
            <th>Alert Level</th>
        </tr>
        {rows_html}
    </table>

    <h2>Interpretation Guide</h2>
    <table>
        <tr><th>PSI Range</th><th>Interpretation</th></tr>
        <tr><td>&lt; 0.05</td><td>No significant drift</td></tr>
        <tr class="caution"><td>0.05 – 0.15</td><td>Moderate drift — monitor closely</td></tr>
        <tr class="warning"><td>0.15 – 0.25</td><td>Significant drift — consider retraining</td></tr>
        <tr class="critical"><td>&gt; 0.25</td><td>Critical drift — retrain immediately</td></tr>
    </table>
</body>
</html>
"""
    output_file.write_text(html, encoding="utf-8")
