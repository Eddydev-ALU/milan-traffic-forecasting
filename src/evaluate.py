"""Forecast accuracy metrics.

All metrics are computed in **original activity units** after inverting the
scaler. MAPE is reported twice on purpose: once over every test point and once
restricted to points above a traffic threshold. Night-time traffic in this
dataset approaches zero, so an unrestricted MAPE is dominated by a handful of
4 a.m. observations and says almost nothing about model quality. Reporting
both, and saying so in the report, is the honest treatment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """Mean absolute percentage error over points with ``y_true > threshold``."""
    mask = y_true > max(threshold, 1e-8)
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted APE: total absolute error as a % of total actual traffic.

    Scale-free like MAPE but immune to near-zero denominators, which makes it
    the metric to use when comparing across areas of very different size.
    """
    denom = np.sum(np.abs(y_true))
    if denom <= 0:
        return float("nan")
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / denom)


def mase(y_true: np.ndarray, y_pred: np.ndarray, train_series: np.ndarray,
         seasonal_period: int = 144) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE.

    MASE < 1 means the model beats a seasonal-naive forecaster on the training
    data's own scale. It is the cleanest single number for cross-area comparison.
    """
    train_series = np.asarray(train_series, dtype=np.float64)
    if len(train_series) <= seasonal_period:
        return float("nan")
    scale = np.mean(np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period]))
    if scale <= 0:
        return float("nan")
    return float(mae(y_true, y_pred) / scale)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_series: np.ndarray | None = None,
    seasonal_period: int = 144,
    mape_threshold: float | None = None,
) -> dict:
    """Full metric bundle for one (model, area) pair."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if mape_threshold is None:
        # 10th percentile of the actual test traffic: excludes the deep-night
        # regime where percentage error is meaningless.
        mape_threshold = float(np.percentile(y_true, 10))
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_all_%": mape(y_true, y_pred, 0.0),
        f"MAPE_above_p10_%": mape(y_true, y_pred, mape_threshold),
        "sMAPE_%": smape(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "mape_threshold": mape_threshold,
    }
    if train_series is not None:
        out["MASE"] = mase(y_true, y_pred, train_series, seasonal_period)
    return out


def metrics_table(results: dict[str, dict], decimals: int = 3) -> pd.DataFrame:
    """Turn ``{model_name: metrics_dict}`` into a report-ready table."""
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    drop = [c for c in ("mape_threshold",) if c in df.columns]
    return df.drop(columns=drop).round(decimals)


# ---------------------------------------------------------------------------
# Failure analysis helpers
# ---------------------------------------------------------------------------

def error_profile(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per-timestep error with calendar context, for the failure analysis."""
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": y_true,
            "predicted": y_pred,
            "error": err,
            "abs_error": np.abs(err),
            "hour": timestamps.hour,
            "dayofweek": timestamps.dayofweek,
            "is_weekend": timestamps.dayofweek >= 5,
        }
    )


def lag_diagnostic(y_true: np.ndarray, y_pred: np.ndarray, max_shift: int = 4) -> dict:
    """Detect the 'model has learned persistence' failure mode.

    If shifting the prediction backwards in time *reduces* the error, the model
    is reproducing the previous observation rather than anticipating the next
    one. Reporting the shift that minimises MAE makes that claim quantitative
    instead of a hand-wave at a plot.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    scores = {}
    for s in range(0, max_shift + 1):
        if s == 0:
            scores[s] = mae(y_true, y_pred)
        else:
            scores[s] = mae(y_true[:-s], y_pred[s:])
    best = min(scores, key=scores.get)
    return {
        "mae_by_backshift": scores,
        "best_backshift": best,
        "improvement_vs_no_shift": scores[0] - scores[best],
        "verdict": (
            "predictions lag the signal (persistence-like behaviour)"
            if best > 0 else "no systematic lag detected"
        ),
    }
