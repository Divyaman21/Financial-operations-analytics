"""K-Means on behavioral features; compare to RFM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def kmeans_segmentation(
    customers: pd.DataFrame,
    feature_cols: list[str],
    k_min: int = 3,
    k_max: int = 8,
    seed: int = 42,
) -> tuple[pd.Series, dict]:
    df = customers.dropna(subset=feature_cols).copy()
    X = StandardScaler().fit_transform(df[feature_cols].values)

    best_k = k_min
    best_score = -1.0
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        if len(df) <= k:
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1.0
        scores[k] = float(sil)
        if sil > best_score:
            best_score = sil
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    out = pd.Series(labels, index=df.index, name="kmeans_cluster")
    meta = {"best_k": best_k, "silhouette_by_k": scores, "silhouette_chosen": float(best_score)}
    return out, meta


def attach_clusters(customers: pd.DataFrame, cluster_series: pd.Series) -> pd.DataFrame:
    df = customers.copy()
    df.loc[cluster_series.index, "kmeans_cluster"] = cluster_series.values
    df["kmeans_cluster"] = df["kmeans_cluster"].fillna(-1).astype(int)
    return df


def cluster_rfm_crosstab(customers: pd.DataFrame) -> pd.DataFrame:
    sub = customers.dropna(subset=["rfm_segment", "kmeans_cluster"])
    ct = pd.crosstab(sub["kmeans_cluster"], sub["rfm_segment"], normalize="index")
    return ct
