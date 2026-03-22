"""Star-schema-friendly exports for Power BI / Tableau."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_bi_tables(
    customers_dim: pd.DataFrame,
    orders_fact: pd.DataFrame,
    model_scores_fact: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    customers_dim.to_csv(out_dir / "customers_dim.csv", index=False)
    orders_fact.to_csv(out_dir / "orders_fact.csv", index=False)
    model_scores_fact.to_csv(out_dir / "model_scores_fact.csv", index=False)
    customers_dim.to_parquet(out_dir / "customers_dim.parquet", index=False)
    orders_fact.to_parquet(out_dir / "orders_fact.parquet", index=False)
    model_scores_fact.to_parquet(out_dir / "model_scores_fact.parquet", index=False)
