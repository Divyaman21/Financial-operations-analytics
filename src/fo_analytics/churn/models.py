"""Churn classifiers: classical + boosted trees with CV tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fo_analytics.config import RANDOM_SEED, TRAIN_TEST_CHURN


# Lagged snapshot columns (see build_customer_features churn_feature_lag_days).
FEATURE_COLS = [
    "lag_frequency_365d",
    "lag_monetary_365d",
    "lag_tenure_days",
    "lag_avg_interpurchase_days",
    "lag_clv_proxy",
]


@dataclass
class ChurnModelResult:
    name: str
    auc_roc: float
    pr_auc: float
    best_params: dict[str, Any] | None
    y_test: np.ndarray
    prob_test: np.ndarray


def _eval_split(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None,
    test_size: float,
    seed: int,
) -> tuple:
    if sample_weight is not None:
        return train_test_split(
            X,
            y,
            sample_weight,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X_tr, X_te, y_tr, y_te, None, None


def train_churn_models(
    customers: pd.DataFrame,
    test_size: float = TRAIN_TEST_CHURN,
    seed: int = RANDOM_SEED,
) -> tuple[list[ChurnModelResult], dict[str, Any]]:
    df = customers.dropna(subset=FEATURE_COLS + ["churned"]).copy()
    X = df[FEATURE_COLS]
    y = df["churned"].astype(int)
    w = None
    if "lag_clv_proxy" in df.columns:
        w = np.log1p(df["lag_clv_proxy"].clip(lower=0).values)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = _eval_split(X, y, w, test_size, seed)

    results: list[ChurnModelResult] = []
    summary: dict[str, Any] = {"n_train": int(len(X_tr)), "n_test": int(len(X_te)), "positive_rate": float(y.mean())}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Logistic regression
    pipe_lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed)),
        ]
    )
    fit_kw_lr = {}
    if w_tr is not None:
        fit_kw_lr["clf__sample_weight"] = w_tr
    pipe_lr.fit(X_tr, y_tr, **fit_kw_lr)
    p_lr = pipe_lr.predict_proba(X_te)[:, 1]
    results.append(
        ChurnModelResult(
            "logistic_regression",
            float(roc_auc_score(y_te, p_lr)),
            float(average_precision_score(y_te, p_lr)),
            None,
            y_te.values,
            p_lr,
        )
    )

    # Random forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr, sample_weight=w_tr if w_tr is not None else None)
    p_rf = rf.predict_proba(X_te)[:, 1]
    results.append(
        ChurnModelResult(
            "random_forest",
            float(roc_auc_score(y_te, p_rf)),
            float(average_precision_score(y_te, p_rf)),
            None,
            y_te.values,
            p_rf,
        )
    )

    # HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.06,
        max_iter=200,
        random_state=seed,
    )
    hgb.fit(X_tr, y_tr, sample_weight=w_tr if w_tr is not None else None)
    p_hgb = hgb.predict_proba(X_te)[:, 1]
    results.append(
        ChurnModelResult(
            "hist_gradient_boosting",
            float(roc_auc_score(y_te, p_hgb)),
            float(average_precision_score(y_te, p_hgb)),
            None,
            y_te.values,
            p_hgb,
        )
    )

    # XGBoost — small grid
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            tree_method="hist",
        )
        grid_xgb = GridSearchCV(
            xgb,
            param_grid={
                "max_depth": [3, 6],
                "learning_rate": [0.08],
                "n_estimators": [250],
                "subsample": [0.85],
                "colsample_bytree": [0.85],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        fit_kw = {"sample_weight": w_tr} if w_tr is not None else {}
        grid_xgb.fit(X_tr, y_tr, **fit_kw)
        p_xgb = grid_xgb.predict_proba(X_te)[:, 1]
        results.append(
            ChurnModelResult(
                "xgboost_tuned",
                float(roc_auc_score(y_te, p_xgb)),
                float(average_precision_score(y_te, p_xgb)),
                grid_xgb.best_params_,
                y_te.values,
                p_xgb,
            )
        )
        summary["xgboost_best_params"] = grid_xgb.best_params_
    except Exception as e:
        summary["xgboost_error"] = str(e)

    # LightGBM — small grid
    try:
        from lightgbm import LGBMClassifier

        lgbm = LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        )
        grid_lgb = GridSearchCV(
            lgbm,
            param_grid={
                "num_leaves": [31, 63],
                "learning_rate": [0.08],
                "n_estimators": [250],
                "subsample": [0.85],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        fit_kw_l = {"sample_weight": w_tr} if w_tr is not None else {}
        grid_lgb.fit(X_tr, y_tr, **fit_kw_l)
        p_lgb = grid_lgb.predict_proba(X_te)[:, 1]
        results.append(
            ChurnModelResult(
                "lightgbm_tuned",
                float(roc_auc_score(y_te, p_lgb)),
                float(average_precision_score(y_te, p_lgb)),
                grid_lgb.best_params_,
                y_te.values,
                p_lgb,
            )
        )
        summary["lightgbm_best_params"] = grid_lgb.best_params_
    except Exception as e:
        summary["lightgbm_error"] = str(e)

    # Calibration note (best by AUC)
    best = max(results, key=lambda r: r.auc_roc)
    prob_true, prob_pred = calibration_curve(best.y_test, best.prob_test, n_bins=8, strategy="uniform")
    summary["calibration_best_model"] = best.name
    summary["calibration_bins_mean_pred"] = prob_pred.tolist()
    summary["calibration_bins_true"] = prob_true.tolist()

    return results, summary


def results_to_dataframe(results: list[ChurnModelResult]) -> pd.DataFrame:
    rows = [{"model": r.name, "auc_roc": r.auc_roc, "pr_auc": r.pr_auc} for r in results]
    return pd.DataFrame(rows).sort_values("auc_roc", ascending=False)


def refit_best_and_score(
    customers: pd.DataFrame,
    best_name: str,
    best_params: dict[str, Any] | None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Refit on all rows for BI export scores (same features as training)."""
    df = customers.dropna(subset=FEATURE_COLS + ["churned"]).copy()
    X = df[FEATURE_COLS]
    y = df["churned"].astype(int)
    w = np.log1p(df["lag_clv_proxy"].clip(lower=0).values)
    probs: np.ndarray

    if best_name == "xgboost_tuned":
        from xgboost import XGBClassifier

        params = dict(best_params or {})
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "auc")
        params.setdefault("random_state", seed)
        params.setdefault("tree_method", "hist")
        clf = XGBClassifier(**params)
        clf.fit(X, y, sample_weight=w)
        probs = clf.predict_proba(X)[:, 1]

    elif best_name == "lightgbm_tuned":
        from lightgbm import LGBMClassifier

        params = dict(best_params or {})
        params.setdefault("objective", "binary")
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("class_weight", "balanced")
        params.setdefault("verbose", -1)
        clf = LGBMClassifier(**params)
        clf.fit(X, y, sample_weight=w)
        probs = clf.predict_proba(X)[:, 1]

    else:
        hgb = HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.06,
            max_iter=200,
            random_state=seed,
        )
        hgb.fit(X, y, sample_weight=w)
        probs = hgb.predict_proba(X)[:, 1]

    out = pd.DataFrame({"customer_id": df["customer_id"].values, "churn_prob": probs})
    return out
