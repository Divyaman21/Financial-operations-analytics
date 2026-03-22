"""End-to-end run: data → models → artifacts → dashboard → BI export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fo_analytics.churn.models import refit_best_and_score, results_to_dataframe, train_churn_models
from fo_analytics.config import (
    ARTIFACTS_DIR,
    CHURN_FEATURE_LAG_DAYS,
    FIGURES_DIR,
    FORECAST_HORIZON_WEEKS,
    HOLDOUT_WEEKS,
    RANDOM_SEED,
    TABLES_DIR,
)
from fo_analytics.dashboard.html import build_dashboard
from fo_analytics.data.synthetic import generate_orders
from fo_analytics.evaluation.metrics import weekly_revenue_series
from fo_analytics.export.bi import write_bi_tables
from fo_analytics.features.customers import build_customer_features
from fo_analytics.forecasting.models import forecast_comparison, save_forecast_diagnostics
from fo_analytics.io.metrics_log import append_metrics_log
from fo_analytics.profitability import (
    add_rfm_segments,
    cohort_retention_table,
    monte_carlo_order_profit,
    profit_driver_regression,
)
from fo_analytics.segmentation.clustering import attach_clusters, cluster_rfm_crosstab, kmeans_segmentation
from fo_analytics.survival.analysis import run_survival_analysis


def _plot_forecast_holdout(
    weekly: pd.Series,
    results: list,
    path: Path,
    holdout: int,
) -> None:
    train = weekly.iloc[: -holdout]
    test = weekly.iloc[-holdout:]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train.index, train.values, label="train", color="gray")
    ax.plot(test.index, test.values, label="actual (holdout)", color="black")
    for r in results:
        if np.any(np.isnan(r.y_pred)):
            continue
        ax.plot(test.index, r.y_pred, label=f"{r.name} pred", alpha=0.85)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Weekly revenue — holdout vs models")
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_all(
    orders: pd.DataFrame | None = None,
    artifacts_root: Path | None = None,
) -> None:
    root = artifacts_root or ARTIFACTS_DIR
    fig_dir = root / "figures"
    tbl_dir = root / "tables"
    dash_dir = root / "dashboard"
    bi_out = root / "bi_export"
    metrics_root = root

    for d in (root, fig_dir, tbl_dir, dash_dir, bi_out):
        d.mkdir(parents=True, exist_ok=True)

    if orders is None:
        orders = generate_orders(seed=RANDOM_SEED)

    as_of = pd.to_datetime(orders["order_date"]).max().normalize()
    weekly = weekly_revenue_series(orders)

    fc_results, fc_comp = forecast_comparison(weekly, HOLDOUT_WEEKS)
    save_forecast_diagnostics(fc_results, fig_dir)
    _plot_forecast_holdout(weekly, fc_results, fig_dir / "forecast_holdout.png", HOLDOUT_WEEKS)
    append_metrics_log(
        metrics_root,
        "forecast_comparison",
        {
            "holdout_weeks": HOLDOUT_WEEKS,
            "horizon_target_weeks": FORECAST_HORIZON_WEEKS,
            "table": fc_comp.to_dict(orient="records"),
        },
    )
    fc_comp.to_csv(tbl_dir / "forecast_comparison.csv", index=False)

    customers = build_customer_features(orders, as_of)
    customers = add_rfm_segments(customers)

    clust_series, clust_meta = kmeans_segmentation(
        customers,
        ["recency_days", "frequency_365d", "monetary_365d", "tenure_days"],
        seed=RANDOM_SEED,
    )
    customers = attach_clusters(customers, clust_series)
    append_metrics_log(metrics_root, "kmeans", clust_meta)
    ct = cluster_rfm_crosstab(customers)
    ct.to_csv(tbl_dir / "cluster_rfm_crosstab.csv")

    churn_results, churn_summary = train_churn_models(customers)
    churn_df = results_to_dataframe(churn_results)
    churn_df.to_csv(tbl_dir / "churn_models.csv", index=False)
    append_metrics_log(
        metrics_root,
        "churn",
        {
            **churn_summary,
            "churn_feature_lag_days": CHURN_FEATURE_LAG_DAYS,
            "models": churn_df.to_dict(orient="records"),
        },
    )

    best_row = churn_df.iloc[0]
    best_name = str(best_row["model"])
    if best_name == "xgboost_tuned":
        best_params = churn_summary.get("xgboost_best_params")
    elif best_name == "lightgbm_tuned":
        best_params = churn_summary.get("lightgbm_best_params")
    else:
        best_params = None
    scores = refit_best_and_score(customers, best_name, best_params)

    surv = run_survival_analysis(customers, figures_dir=fig_dir, tables_dir=tbl_dir)
    append_metrics_log(
        metrics_root,
        "survival",
        {"km_plot": surv.km_plot_path, "cox_summary": surv.cox_summary_csv, "ph_test_p": surv.ph_test_p, "notes": surv.notes},
    )

    reg = profit_driver_regression(customers)
    with (tbl_dir / "profit_regression_summary.txt").open("w", encoding="utf-8") as f:
        f.write(reg["summary_text"])
    append_metrics_log(metrics_root, "profit_regression", {k: v for k, v in reg.items() if k != "summary_text"})

    mc = monte_carlo_order_profit(orders, n_draws=5000, n_orders=5000, seed=RANDOM_SEED)
    append_metrics_log(metrics_root, "monte_carlo_orders", mc)

    cohort = cohort_retention_table(orders, as_of)
    cohort.to_csv(tbl_dir / "cohort_retention.csv")

    customers_dim = customers[
        [
            "customer_id",
            "first_purchase",
            "last_purchase",
            "recency_days",
            "frequency_365d",
            "monetary_365d",
            "tenure_days",
            "rfm_segment",
            "kmeans_cluster",
            "churned",
            "clv_proxy",
        ]
    ].copy()
    customers_dim["as_of_date"] = as_of.date().isoformat()

    orders_fact = orders.copy()
    orders_fact["order_date"] = pd.to_datetime(orders_fact["order_date"]).dt.date.astype(str)

    model_scores = customers_dim[["customer_id", "as_of_date", "rfm_segment", "kmeans_cluster"]].merge(
        scores, on="customer_id", how="left"
    )
    model_scores["duration_days"] = customers.set_index("customer_id")["duration_days"].reindex(
        model_scores["customer_id"]
    ).values
    model_scores["event_observed"] = customers.set_index("customer_id")["event_observed"].reindex(
        model_scores["customer_id"]
    ).values
    fc_rank = fc_comp.replace([np.inf, -np.inf], np.nan).dropna(subset=["rmse"])
    best_fc = fc_rank.sort_values("rmse").iloc[0]["model"] if len(fc_rank) else "n/a"
    model_scores["forecast_model_best"] = best_fc
    model_scores["weekly_revenue_last"] = float(weekly.iloc[-1])

    write_bi_tables(customers_dim, orders_fact, model_scores, bi_out)

    build_dashboard(
        dash_dir,
        fc_comp,
        churn_df,
        mc,
        float(reg["rsquared"]),
        fig_dir,
    )

    print("Pipeline complete. Artifacts:", root.resolve())
