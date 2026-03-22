import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def weekly_revenue_series(orders: pd.DataFrame) -> pd.Series:
    """Aggregate order revenue to weekly (Monday week start)."""
    df = orders.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.set_index("order_date")
    weekly = df["revenue"].resample("W-MON").sum()
    weekly = weekly.rename("revenue")
    return weekly.asfreq("W-MON", fill_value=0.0)
