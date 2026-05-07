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
from fo_analytics.registry import ModelRegistry, ModelSerializer
from fo_analytics.registry.serializer import compute_dataset_hash
from fo_analytics.forecasting.ensemble import ForecastEnsemble
from fo_analytics.explainability import SHAPExplainer
from fo_analytics.monitoring import DriftDetector, generate_drift_report
from fo_analytics.alerting import AlertEngine


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

    # Initialize new feature components
    registry = ModelRegistry(manifest_file=root / "model_manifest.json")
    serializer = ModelSerializer(artifact_dir=root / "models")
    drift_detector = DriftDetector(baseline_file=root / "monitoring" / "baseline.json")
    alert_engine = AlertEngine()

    if orders is None:
        orders = generate_orders(seed=RANDOM_SEED)

    as_of = pd.to_datetime(orders["order_date"]).max().normalize()
    weekly = weekly_revenue_series(orders)

    fc_results, fc_comp = forecast_comparison(weekly, HOLDOUT_WEEKS)
    
    # Ensemble feature
    y_test = fc_results[0].y_test.values
    fc_dict = {r.name: r.y_pred for r in fc_results if not np.any(np.isnan(r.y_pred))}
    ensemble = ForecastEnsemble()
    weights = ensemble.fit_weights(y_test, fc_dict, method="inverse_variance")
    y_ens = ensemble.predict(fc_dict)
    
    from fo_analytics.evaluation.metrics import regression_metrics
    from fo_analytics.forecasting.models import ForecastResult
    ens_metrics = regression_metrics(y_test, y_ens)
    fc_results.append(ForecastResult("ensemble", fc_results[0].y_train, fc_results[0].y_test, y_ens, ens_metrics, {"weights": weights}))
    fc_comp = pd.concat([fc_comp, pd.DataFrame([{"model": "ensemble", **ens_metrics}])], ignore_index=True)

    save_forecast_diagnostics(fc_results, fig_dir)
    _plot_forecast_holdout(weekly, fc_results, fig_dir / "forecast_holdout.png", HOLDOUT_WEEKS)
    
    # Save ensemble dummy model to registry
    v_id, p_path = serializer.save_model(weights, "forecast_ensemble", "ensemble", metrics=ens_metrics)
    registry.register("forecast_ensemble", p_path, v_id, ens_metrics, weights, "ensemble")

    # Alert on forecast error
    alert_engine.check_forecast_error(ens_metrics["mape"])
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

    # Data Drift feature
    features_to_monitor = ["recency_days", "frequency_365d", "monetary_365d", "tenure_days", "avg_interpurchase_days", "clv_proxy"]
    if drift_detector.baseline_file.exists():
        drift_results = drift_detector.check_all(customers)
        generate_drift_report(drift_results, root / "monitoring" / "drift_report.html")
        alert_engine.check_data_drift(drift_results["max_psi"])
    else:
        # First run: compute baseline
        baseline = drift_detector.compute_baseline(customers, features_to_monitor)
        drift_detector.save_baseline(baseline)

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
    scores, best_model = refit_best_and_score(customers, best_name, best_params)
    
    # Registry feature (Churn)
    from fo_analytics.churn.models import FEATURE_COLS
    X_train = customers.dropna(subset=FEATURE_COLS + ["churned"])[FEATURE_COLS]
    d_hash = compute_dataset_hash(X_train)
    v_id, p_path = serializer.save_model(best_model, "churn_best", best_name, metrics={"auc_roc": float(best_row["auc_roc"]), "pr_auc": float(best_row["pr_auc"])}, hyperparams=best_params, dataset_hash=d_hash)
    registry.register("churn_best", p_path, v_id, {"auc_roc": float(best_row["auc_roc"]), "pr_auc": float(best_row["pr_auc"])}, best_params or {}, best_name, d_hash)

    # Explainability feature
    try:
        explainer = SHAPExplainer(best_model, X_train)
        explainer.summary_plot(X_train, root / "explainability" / "shap_summary.png")
    except Exception as e:
        print(f"SHAP explainer failed: {e}")
        
    # Alert on churn rate
    churn_rate = float(customers["churned"].mean())
    alert_engine.check_churn_spike(churn_rate, historical_mean=0.20, historical_std=0.05)

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

    # Alert on margin & Summary
    margin = float(mc["mean_total_profit"] / orders["revenue"].sum()) if orders["revenue"].sum() > 0 else 0.0
    alert_engine.check_margin_threshold(margin)
    alert_engine.generate_report(root / "alerts.html")
    alert_engine.save_json(root / "alerts.json")

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
