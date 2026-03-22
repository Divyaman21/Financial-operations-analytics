"""Kaplan–Meier and Cox PH for time-to-churn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test


@dataclass
class SurvivalArtifacts:
    km_plot_path: str | None
    cox_summary_csv: str | None
    ph_test_p: float | None
    notes: str


def run_survival_analysis(
    customers: pd.DataFrame,
    segment_col: str = "rfm_segment",
    figures_dir: Any = None,
    tables_dir: Any = None,
) -> SurvivalArtifacts:
    df = customers.dropna(subset=["duration_days", "event_observed"]).copy()
    if len(df) < 40:
        return SurvivalArtifacts(None, None, None, "insufficient_rows")

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 4))
    kmf.fit(df["duration_days"], df["event_observed"], label="All customers")
    kmf.plot_survival_function(ax=ax, color="steelblue")
    ax.set_title("Kaplan–Meier: time to churn (from first purchase)")
    ax.set_xlabel("Days")
    fig.tight_layout()
    km_path = None
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        km_path = str(figures_dir / "kaplan_meier_overall.png")
        fig.savefig(km_path, dpi=150)
    plt.close(fig)

    # By segment if present
    if segment_col in df.columns and df[segment_col].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        for seg, sub in df.groupby(segment_col):
            if len(sub) < 10:
                continue
            k = KaplanMeierFitter()
            k.fit(sub["duration_days"], sub["event_observed"], label=str(seg))
            k.plot_survival_function(ax=ax)
        ax.set_title("Kaplan–Meier by RFM segment")
        ax.set_xlabel("Days")
        fig.tight_layout()
        if figures_dir is not None:
            fig.savefig(figures_dir / "kaplan_meier_by_segment.png", dpi=150)
        plt.close(fig)

    # Cox PH on scaled covariates (lifelines expects no NaN)
    covars = ["recency_days", "frequency_365d", "monetary_365d", "tenure_days"]
    cox_df = df[["duration_days", "event_observed"] + [c for c in covars if c in df.columns]].dropna()
    cox_path = None
    ph_p = None
    notes = ""
    if len(cox_df) >= 60 and len(covars) >= 2:
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(cox_df, duration_col="duration_days", event_col="event_observed")
            if tables_dir is not None:
                tables_dir.mkdir(parents=True, exist_ok=True)
                cox_path = str(tables_dir / "cox_summary.csv")
                cph.summary.to_csv(cox_path)
            try:
                results = proportional_hazard_test(cph, cox_df)
                ph_p = float(np.nanmin(results._p_value))
            except Exception:
                ph_p = None
        except Exception as e:
            notes = f"cox_fit_failed:{e}"
    else:
        notes = "cox_skipped_small_sample"

    return SurvivalArtifacts(km_path, cox_path, ph_p, notes)
