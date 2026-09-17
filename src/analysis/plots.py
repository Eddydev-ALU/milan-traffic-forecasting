"""Figures for Section 4: actual-vs-predicted, error profiles, failure cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eda import _save

MODEL_COLOURS = {
    "SARIMAX": "#c1543a",
    "LSTM": "#3b6ea5",
    "TCN": "#4c9a52",
    "Persistence": "#8a8a8a",
}


def _colour(model_name: str) -> str:
    for key, col in MODEL_COLOURS.items():
        if model_name.upper().startswith(key.upper()):
            return col
    return "#7a4fa3"


def plot_actual_vs_predicted(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    square_id: int,
    out_dir: Path,
    metrics: dict | None = None,
) -> Path:
    """One of the nine required superposed plots (3 models x 3 areas)."""
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(timestamps, y_true, lw=1.0, color="#22405e", label="actual")
    ax.plot(timestamps, y_pred, lw=1.0, color=_colour(model_name), ls="--", label="predicted")
    title = f"Square {square_id} - {model_name} - one-step-ahead, 16-22 Dec 2013"
    if metrics:
        title += (f"   (MAE {metrics['MAE']:,.1f} | RMSE {metrics['RMSE']:,.1f}"
                  f" | WAPE {metrics['WAPE_%']:.1f}%)")
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_ylabel("Internet activity")
    ax.legend(frameon=False, ncol=2)

    ax2.fill_between(timestamps, 0, np.asarray(y_pred) - np.asarray(y_true),
                     color=_colour(model_name), alpha=0.5, lw=0)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_ylabel("error")
    ax2.set_xlabel("date")
    for day in pd.date_range(timestamps[0].normalize(), timestamps[-1], freq="D"):
        if day.dayofweek >= 5:
            for a in (ax, ax2):
                a.axvspan(day, day + pd.Timedelta(days=1), color="grey", alpha=0.12, lw=0)
    fig.tight_layout()
    safe = model_name.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
    return _save(fig, out_dir, f"fig_pred_sq{square_id}_{safe}")


def plot_model_overlay(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    square_id: int,
    out_dir: Path,
    zoom: tuple[str, str] | None = None,
) -> Path:
    """All models on one axis -- the figure that carries the comparison."""
    ts = pd.DatetimeIndex(timestamps)
    sel = slice(None)
    if zoom:
        tz = ts.tz
        lo = pd.Timestamp(zoom[0]).tz_localize(tz)
        hi = pd.Timestamp(zoom[1]).tz_localize(tz)
        sel = (ts >= lo) & (ts <= hi)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(ts[sel], np.asarray(y_true)[sel], lw=1.6, color="#22405e", label="actual", zorder=5)
    for name, pred in predictions.items():
        ax.plot(ts[sel], np.asarray(pred)[sel], lw=1.0, ls="--",
                color=_colour(name), label=name, alpha=0.9)
    ax.set_title(f"Square {square_id} - all models"
                 + (f" (zoom {zoom[0]} to {zoom[1]})" if zoom else ""), loc="left")
    ax.set_ylabel("Internet activity")
    ax.set_xlabel("date")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    suffix = "_zoom" if zoom else ""
    return _save(fig, out_dir, f"fig_overlay_sq{square_id}{suffix}")


def plot_error_breakdown(profiles: dict[str, pd.DataFrame], square_id: int,
                         out_dir: Path) -> Path:
    """Where does each model fail? By hour, by weekday, and over the week."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for name, prof in profiles.items():
        by_hour = prof.groupby("hour")["abs_error"].mean()
        axes[0].plot(by_hour.index, by_hour.values, lw=1.2, color=_colour(name), label=name)
        by_dow = prof.groupby("dayofweek")["abs_error"].mean()
        axes[1].plot(by_dow.index, by_dow.values, lw=1.2, marker="o", ms=3,
                     color=_colour(name), label=name)
        axes[2].plot(prof["timestamp"], prof["abs_error"].rolling(18, min_periods=1).mean(),
                     lw=0.9, color=_colour(name), label=name, alpha=0.85)

    axes[0].set_xlabel("hour of day")
    axes[0].set_ylabel("mean |error|")
    axes[0].set_title("(a) Error by hour")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_title("(b) Error by weekday")
    axes[2].set_title("(c) 3-hour rolling mean |error|")
    axes[2].set_xlabel("date")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Error decomposition, square {square_id}", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig_error_breakdown_sq{square_id}")


def plot_training_curves(histories: dict[str, object], out_dir: Path) -> Path:
    """Learning curves -- evidence for the early-stopping and tuning narrative."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, hist in histories.items():
        if not getattr(hist, "train_loss", None):
            continue
        axes[0].plot(hist.train_loss, lw=1.2, color=_colour(name), label=name)
        if hist.val_mae_raw:
            axes[1].plot(hist.val_mae_raw, lw=1.2, color=_colour(name), label=name)
            if hist.best_epoch >= 0:
                axes[1].scatter([hist.best_epoch], [hist.best_val_mae],
                                color=_colour(name), s=28, zorder=5)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss (scaled)")
    axes[0].set_title("(a) Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation MAE (activity units)")
    axes[1].set_title("(b) Validation MAE, dot = selected epoch")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_training_curves")


def plot_metric_comparison(table: pd.DataFrame, out_dir: Path,
                           metric: str = "MAE", name: str = "fig_metric_comparison") -> Path:
    """Grouped bars: metric by model, across the three areas."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    pivot = table.pivot_table(index="model", columns="square_id", values=metric)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[col].values, width, label=f"square {col}")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by model and area (test week)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, name)
