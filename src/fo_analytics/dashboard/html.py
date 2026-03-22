"""Static executive HTML dashboard from artifacts."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _embed_png(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;" alt="{path.name}"/>'


def build_dashboard(
    out_dir: Path,
    forecast_table: pd.DataFrame,
    churn_table: pd.DataFrame,
    mc_summary: dict,
    regression_r2: float,
    figures_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    churn_plot_path = out_dir / "_churn_bar.png"
    if not churn_table.empty:
        churn_table.sort_values("auc_roc").plot.barh(x="model", y="auc_roc", ax=ax, legend=False, color="teal")
        ax.set_title("Churn models — holdout AUC-ROC")
    fig.tight_layout()
    fig.savefig(churn_plot_path, dpi=120)
    plt.close(fig)

    fc_html = forecast_table.to_html(classes="tbl", border=0, index=False)
    ch_html = churn_table.to_html(classes="tbl", border=0, index=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Financial Operations — Executive Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }}
    h1 {{ font-size: 1.6rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
    .card {{ background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    table.tbl {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
    table.tbl th, table.tbl td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; }}
    .kpi {{ font-size: 1.4rem; font-weight: 600; }}
    .note {{ font-size: 0.85rem; color: #555; }}
  </style>
</head>
<body>
  <h1>Financial Operations Analytics</h1>
  <p class="note">Static snapshot generated from the Python pipeline. Forecasts use a 26-week holdout; churn metrics are stratified holdout.</p>

  <div class="grid">
    <div class="card">
      <h2>Profit driver regression (log revenue)</h2>
      <p class="kpi">R² = {regression_r2:.3f}</p>
      <p class="note">OLS on frequency, recency, tenure — see academic report for full diagnostics.</p>
    </div>
    <div class="card">
      <h2>Monte Carlo — order profit</h2>
      <p class="kpi">Mean sim profit: {mc_summary.get("mean_total_profit", float("nan")):,.0f}</p>
      <p class="note">p5–p95: {mc_summary.get("p5", float("nan")):,.0f} — {mc_summary.get("p95", float("nan")):,.0f} ({mc_summary.get("n_orders_in_sim", "?")} orders, {mc_summary.get("n_draws", "?")} draws)</p>
    </div>
  </div>

  <h2>Revenue forecast model comparison</h2>
  <div class="card">{fc_html}</div>
  {_embed_png(figures_dir / "forecast_holdout.png")}

  <h2>Churn models</h2>
  <div class="card">{ch_html}</div>
  {_embed_png(churn_plot_path)}
  {_embed_png(figures_dir / "kaplan_meier_overall.png")}

  <h2>Residual diagnostics (examples)</h2>
  <div class="grid">
    <div class="card">{_embed_png(figures_dir / "residuals_holt_winters.png")}</div>
    <div class="card">{_embed_png(figures_dir / "residuals_sarima.png")}</div>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
