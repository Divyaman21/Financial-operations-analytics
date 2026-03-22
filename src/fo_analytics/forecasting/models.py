"""Revenue forecasting: Holt–Winters, SARIMAX, Prophet with diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from fo_analytics.evaluation.metrics import regression_metrics

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ForecastResult:
    name: str
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    metrics: dict[str, float]
    extras: dict[str, Any]


def split_series(y: pd.Series, holdout: int) -> tuple[pd.Series, pd.Series]:
    y = y.dropna()
    if len(y) <= holdout + 8:
        raise ValueError("Series too short for holdout")
    train, test = y.iloc[:-holdout], y.iloc[-holdout:]
    return train, test


def fit_predict_sarima(
    train: pd.Series, horizon: int, seasonal_period: int = 52
) -> tuple[np.ndarray, Any, dict[str, float]]:
    """Small seasonal grid; pick by BIC on training sample."""
    train = train.astype(float)
    sp = min(seasonal_period, max(2, len(train) // 3))
    best_bic = np.inf
    best_res = None
    best_order = None
    best_sorder = None

    orders = [(0, 1, 1), (1, 1, 1), (1, 0, 1), (2, 1, 1)]
    seasonal_orders = [(0, 0, 0, 0), (1, 0, 1, sp)]
    for order in orders:
        for sorder in seasonal_orders:
            try:
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=sorder,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                if res.bic < best_bic:
                    best_bic = res.bic
                    best_res = res
                    best_order = order
                    best_sorder = sorder
            except Exception:
                continue

    if best_res is None:
        model = SARIMAX(train, order=(0, 1, 1), seasonal_order=(0, 0, 0, 0))
        best_res = model.fit(disp=False)
        best_order = (0, 1, 1)
        best_sorder = (0, 0, 0, 0)
        best_bic = float(best_res.bic)

    fc = best_res.get_forecast(steps=horizon)
    pred = fc.predicted_mean.values
    ic = {"aic": float(best_res.aic), "bic": float(best_res.bic), "hqic": float(best_res.hqic)}
    extras = {"order": best_order, "seasonal_order": best_sorder, "info_criteria": ic}
    return pred, best_res, extras


def ljung_box_summary(residuals: np.ndarray, lags: int = 10) -> dict[str, Any]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < lags + 2:
        return {"ljung_box_min_p": None, "lags": lags}
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p = float(lb["lb_pvalue"].iloc[0])
    return {"ljung_box_p_at_lag": p, "lags": lags, "pass_0_05": bool(p > 0.05)}


def run_prophet(train: pd.Series, horizon: int, holidays_df: pd.DataFrame | None) -> tuple[np.ndarray, Any]:
    from prophet import Prophet

    df = train.reset_index()
    df.columns = ["ds", "y"]
    kw: dict[str, Any] = {
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05,
    }
    if holidays_df is not None and not holidays_df.empty:
        kw["holidays"] = holidays_df
    m = Prophet(**kw)
    m.add_country_holidays(country_name="US")
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="W-MON")
    fcst = m.predict(future)
    yhat = fcst["yhat"].values[-horizon:]
    return yhat, m


def forecast_comparison(
    weekly_revenue: pd.Series,
    holdout_weeks: int,
    holiday_dates: list[pd.Timestamp] | None = None,
) -> tuple[list[ForecastResult], pd.DataFrame]:
    train, test = split_series(weekly_revenue, holdout_weeks)
    horizon = len(test)
    results: list[ForecastResult] = []

    # Holt-Winters
    try:
        seasonal = min(52, max(4, len(train) // 2))
        if len(train) < 2 * seasonal:
            seasonal = None
        hw = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if seasonal else None,
            seasonal_periods=seasonal,
            initialization_method="estimated",
        )
        hw_fit = hw.fit(optimized=True)
        y_hw = hw_fit.forecast(horizon).values
        res_hw = train.values - hw_fit.fittedvalues.values
        extras_hw = {
            "ljung_box": ljung_box_summary(res_hw),
            "residual_std": float(np.nanstd(res_hw)),
        }
        results.append(
            ForecastResult(
                "holt_winters",
                train,
                test,
                y_hw,
                regression_metrics(test.values, y_hw),
                extras_hw,
            )
        )
    except Exception as e:
        results.append(
            ForecastResult(
                "holt_winters",
                train,
                test,
                np.full(horizon, np.nan),
                {"mape": np.nan, "smape": np.nan, "rmse": np.nan},
                {"error": str(e)},
            )
        )

    # SARIMAX
    try:
        y_s, sarima_res, sarima_extras = fit_predict_sarima(train, horizon)
        res_s = sarima_res.resid.values[-len(train) :]
        sarima_extras["ljung_box"] = ljung_box_summary(res_s)
        results.append(
            ForecastResult(
                "sarima",
                train,
                test,
                y_s,
                regression_metrics(test.values, y_s),
                sarima_extras,
            )
        )
    except Exception as e:
        results.append(
            ForecastResult(
                "sarima",
                train,
                test,
                np.full(horizon, np.nan),
                {"mape": np.nan, "smape": np.nan, "rmse": np.nan},
                {"error": str(e)},
            )
        )

    # Prophet
    try:
        holidays_df = None
        if holiday_dates:
            holidays_df = pd.DataFrame({"ds": pd.to_datetime(holiday_dates), "holiday": "custom"})
        y_p, _prophet_model = run_prophet(train, horizon, holidays_df)
        results.append(
            ForecastResult(
                "prophet",
                train,
                test,
                y_p,
                regression_metrics(test.values, y_p),
                {"note": "Ljung-Box on Prophet residuals omitted; see residual plot in artifacts"},
            )
        )
    except Exception as e:
        results.append(
            ForecastResult(
                "prophet",
                train,
                test,
                np.full(horizon, np.nan),
                {"mape": np.nan, "smape": np.nan, "rmse": np.nan},
                {"error": str(e)},
            )
        )

    rows = []
    for r in results:
        row = {"model": r.name, **r.metrics}
        if "info_criteria" in r.extras:
            row["aic"] = r.extras["info_criteria"].get("aic")
            row["bic"] = r.extras["info_criteria"].get("bic")
        if "ljung_box" in r.extras and isinstance(r.extras["ljung_box"], dict):
            row["ljung_box_p"] = r.extras["ljung_box"].get("ljung_box_p_at_lag")
        rows.append(row)
    comp = pd.DataFrame(rows)
    return results, comp


def save_forecast_diagnostics(
    results: list[ForecastResult],
    figures_dir: Any,
) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        if r.name == "prophet" or len(r.y_pred) != len(r.y_test):
            continue
        resid = r.y_test.values - r.y_pred
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(resid, color="steelblue")
        ax.set_title(f"{r.name} holdout residuals")
        ax.set_xlabel("Week index (holdout)")
        fig.tight_layout()
        fig.savefig(figures_dir / f"residuals_{r.name}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        qqplot(resid, line="45", ax=ax)
        ax.set_title(f"{r.name} residual QQ")
        fig.tight_layout()
        fig.savefig(figures_dir / f"qq_{r.name}.png", dpi=150)
        plt.close(fig)
