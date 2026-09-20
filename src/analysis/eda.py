"""Exploratory and time-series analysis.

Every function returns both the figure/table *and* the numbers behind it, so
the report can quote statistics rather than pointing vaguely at a plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2.1 Spatial distribution of total traffic
# ---------------------------------------------------------------------------

def total_traffic_per_square(matrix: np.ndarray) -> np.ndarray:
    """Sum over the whole observation window, computed in blocks.

    Blocking keeps the working set small when ``matrix`` is memory-mapped: we
    never force the full 341 MiB array into RAM at once.
    """
    n = matrix.shape[0]
    totals = np.empty(n, dtype=np.float64)
    step = 1000
    for i in range(0, n, step):
        totals[i: i + step] = np.asarray(matrix[i: i + step], dtype=np.float64).sum(axis=1)
    return totals


def top_k_squares(totals: np.ndarray, square_ids: np.ndarray, k: int = 3) -> list[int]:
    return [int(square_ids[i]) for i in np.argsort(totals)[::-1][:k]]


def describe_distribution(totals: np.ndarray) -> dict:
    """Quantify the skew and concentration the histogram shows visually."""
    pos = totals[totals > 0]
    order = np.sort(totals)[::-1]
    csum = np.cumsum(order) / order.sum()
    # Gini coefficient of the spatial traffic distribution.
    s = np.sort(totals)
    n = len(s)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum())) if s.sum() > 0 else np.nan
    return {
        "n_squares": int(len(totals)),
        "n_zero_squares": int((totals <= 0).sum()),
        "mean": float(totals.mean()),
        "median": float(np.median(totals)),
        "std": float(totals.std()),
        "skewness": float(pd.Series(totals).skew()),
        "kurtosis": float(pd.Series(totals).kurtosis()),
        "max_over_median": float(totals.max() / np.median(totals)),
        "share_top_1pct": float(csum[int(0.01 * len(totals))]),
        "share_top_5pct": float(csum[int(0.05 * len(totals))]),
        "share_top_10pct": float(csum[int(0.10 * len(totals))]),
        "gini": gini,
        "log10_range": float(np.log10(pos.max() / pos.min())) if len(pos) else np.nan,
    }


def plot_spatial_distribution(totals: np.ndarray, out_dir: Path,
                              grid_size: int = 100) -> tuple[Path, dict]:
    """Histogram (log scale), ECDF and a 100x100 map of the Milan grid."""
    stats = describe_distribution(totals)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    pos = totals[totals > 0]
    axes[0].hist(np.log10(pos), bins=60, color="#3b6ea5", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("log$_{10}$(total Internet activity)")
    axes[0].set_ylabel("number of squares")
    axes[0].set_title("(a) Distribution across 10,000 areas")
    axes[0].axvline(np.log10(np.median(pos)), color="crimson", ls="--", lw=1,
                    label=f"median = {np.median(pos):,.0f}")
    axes[0].legend(frameon=False, fontsize=8)

    srt = np.sort(totals)
    axes[1].plot(srt, np.arange(1, len(srt) + 1) / len(srt), color="#3b6ea5")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("total Internet activity (log)")
    axes[1].set_ylabel("empirical CDF")
    axes[1].set_title("(b) ECDF")

    if grid_size * grid_size == totals.size:
        grid = totals.reshape(grid_size, grid_size)
        im = axes[2].imshow(np.log10(grid + 1), origin="lower", cmap="magma")
        axes[2].set_title("(c) Spatial layout (log scale)")
        axes[2].set_xlabel("grid column")
        axes[2].set_ylabel("grid row")
        axes[2].grid(False)
        fig.colorbar(im, ax=axes[2], fraction=0.046, label="log$_{10}$(total + 1)")
    else:
        # Non-square grid (e.g. the synthetic smoke-test subset): fall back to a
        # Lorenz curve, which shows the same concentration story.
        share = np.cumsum(np.sort(totals)) / totals.sum()
        frac = np.arange(1, len(totals) + 1) / len(totals)
        axes[2].plot(frac, share, color="#3b6ea5")
        axes[2].plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
        axes[2].set_xlabel("cumulative share of areas")
        axes[2].set_ylabel("cumulative share of traffic")
        axes[2].set_title("(c) Lorenz curve of traffic concentration")

    fig.suptitle("Total Internet traffic per geographical area, Nov 2013 - Jan 2014", y=1.03)
    return _save(fig, out_dir, "fig01_spatial_distribution"), stats


# ---------------------------------------------------------------------------
# 2.2 Time series of selected squares
# ---------------------------------------------------------------------------

def plot_selected_series(
    matrix: np.ndarray,
    square_ids: np.ndarray,
    timestamps: pd.DatetimeIndex,
    selected: list[int],
    start: str,
    end: str,
    out_dir: Path,
    labels: dict[int, str] | None = None,
) -> Path:
    """Stacked time series for the required five squares over two weeks."""
    tz = timestamps.tz
    lo = pd.Timestamp(start).tz_localize(tz)
    hi = pd.Timestamp(end).tz_localize(tz)
    sel = (timestamps >= lo) & (timestamps <= hi)
    ts = timestamps[sel]

    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 2.1 * len(selected)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sid in zip(axes, selected):
        pos = int(np.flatnonzero(square_ids == sid)[0])
        series = np.asarray(matrix[pos][sel], dtype=np.float64)
        ax.plot(ts, series, lw=0.7, color="#22405e")
        tag = f" - {labels[sid]}" if labels and sid in labels else ""
        ax.set_title(f"Square {sid}{tag}   (total over window: {series.sum():,.0f})",
                     loc="left", fontsize=9)
        ax.set_ylabel("activity")
        # Shade weekends to make the weekly cycle unmistakable.
        for day in pd.date_range(lo, hi, freq="D"):
            if day.dayofweek >= 5:
                ax.axvspan(day, day + pd.Timedelta(days=1), color="grey", alpha=0.12, lw=0)
    axes[-1].set_xlabel("date")
    fig.suptitle("Internet activity, first two weeks (weekends shaded)", y=0.995)
    fig.tight_layout()
    return _save(fig, out_dir, "fig02_selected_series")


def compare_series_statistics(
    matrix: np.ndarray, square_ids: np.ndarray, timestamps: pd.DatetimeIndex,
    selected: list[int], start: str, end: str
) -> pd.DataFrame:
    """Numbers to anchor the 'similarities and differences' discussion."""
    tz = timestamps.tz
    sel = ((timestamps >= pd.Timestamp(start).tz_localize(tz))
           & (timestamps <= pd.Timestamp(end).tz_localize(tz)))
    ts = timestamps[sel]
    rows = []
    for sid in selected:
        pos = int(np.flatnonzero(square_ids == sid)[0])
        s = pd.Series(np.asarray(matrix[pos][sel], dtype=np.float64), index=ts)
        daily = s.groupby(s.index.dayofweek).mean()
        weekday = daily.loc[daily.index <= 4].mean()
        weekend = daily.loc[daily.index >= 5].mean()
        night = s[(s.index.hour >= 2) & (s.index.hour < 5)].mean()
        peak_hour = s.groupby(s.index.hour).mean().idxmax()
        rows.append({
            "square_id": sid,
            "mean": s.mean(),
            "std": s.std(),
            "cv": s.std() / s.mean() if s.mean() else np.nan,
            "max/mean (burstiness)": s.max() / s.mean() if s.mean() else np.nan,
            "peak_hour": int(peak_hour),
            "weekday_mean": weekday,
            "weekend_mean": weekend,
            "weekend/weekday": weekend / weekday if weekday else np.nan,
            "night_mean(02-05)": night,
            "night/peak": night / s.max() if s.max() else np.nan,
        })
    return pd.DataFrame(rows).set_index("square_id").round(3)


# ---------------------------------------------------------------------------
# 2.3 Time-series characterisation of the busiest square
# ---------------------------------------------------------------------------

def plot_acf_pacf(series: np.ndarray, out_dir: Path, max_lag: int = 1100,
                  name: str = "fig03_acf_pacf") -> tuple[Path, dict]:
    """ACF out to beyond one week plus PACF over the short range.

    This is the figure that justifies the input window length ``L`` used by the
    neural models: read off where the autocorrelation stops being informative
    and where the daily (144) and weekly (1008) peaks sit.
    """
    from statsmodels.tsa.stattools import acf, pacf

    s = np.asarray(series, dtype=np.float64)
    a = acf(s, nlags=max_lag, fft=True)
    p = pacf(s, nlags=60, method="ywm")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].plot(np.arange(len(a)), a, lw=0.8, color="#22405e")
    for lag, lbl in ((144, "1 day"), (288, "2 days"), (1008, "1 week")):
        if lag < len(a):
            axes[0].axvline(lag, color="crimson", ls=":", lw=1)
            axes[0].text(lag, 0.95 * a.max(), lbl, rotation=90, fontsize=7,
                         color="crimson", va="top", ha="right")
    axes[0].set_title("(a) ACF to one week+")
    axes[0].set_xlabel("lag (10-min steps)")
    axes[0].set_ylabel("autocorrelation")

    axes[1].stem(np.arange(len(a[:145])), a[:145], markerfmt=" ", basefmt=" ")
    axes[1].set_title("(b) ACF, first day")
    axes[1].set_xlabel("lag")

    axes[2].stem(np.arange(len(p)), p, markerfmt=" ", basefmt=" ")
    conf = 1.96 / np.sqrt(len(s))
    axes[2].axhline(conf, color="crimson", ls="--", lw=0.8)
    axes[2].axhline(-conf, color="crimson", ls="--", lw=0.8)
    axes[2].set_title("(c) PACF, first 60 lags")
    axes[2].set_xlabel("lag")
    fig.tight_layout()

    sig = np.flatnonzero(np.abs(p) > conf)
    stats = {
        "acf_lag1": float(a[1]),
        "acf_lag6_1h": float(a[6]),
        "acf_lag144_1day": float(a[144]) if len(a) > 144 else np.nan,
        "acf_lag1008_1week": float(a[1008]) if len(a) > 1008 else np.nan,
        "first_lag_acf_below_0.5": int(np.argmax(a < 0.5)) if (a < 0.5).any() else None,
        "significant_pacf_lags": sig.tolist()[:12],
        "pacf_conf_bound": float(conf),
    }
    return _save(fig, out_dir, name), stats


def stationarity_tests(series: np.ndarray, seasonal_period: int = 144) -> pd.DataFrame:
    """ADF and KPSS on the raw and seasonally-differenced series.

    The two tests have opposite null hypotheses (ADF: unit root; KPSS:
    stationarity). Reporting them together is far more convincing than either
    alone -- agreement is strong evidence, disagreement flags a near-unit-root
    or trend-stationary case worth discussing.
    """
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    s = np.asarray(series, dtype=np.float64)
    variants = {
        "raw": s,
        "log1p": np.log1p(np.clip(s, 0, None)),
        f"seasonal diff (m={seasonal_period})": s[seasonal_period:] - s[:-seasonal_period],
        "first diff": np.diff(s),
    }
    rows = []
    for label, x in variants.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
            kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
        rows.append({
            "series": label,
            "ADF stat": adf_stat,
            "ADF p": adf_p,
            "ADF: reject unit root (5%)": adf_p < 0.05,
            "KPSS stat": kpss_stat,
            "KPSS p": kpss_p,
            "KPSS: reject stationarity (5%)": kpss_p < 0.05,
        })
    return pd.DataFrame(rows).set_index("series").round(4)


def plot_decomposition(series: np.ndarray, timestamps: pd.DatetimeIndex, out_dir: Path,
                       periods=(144, 1008), name: str = "fig04_mstl") -> tuple[Path, dict]:
    """MSTL: trend + daily seasonal + weekly seasonal + remainder.

    Two seasonalities are separated simultaneously, which single-period STL
    cannot do. The variance shares quantify how much of the signal is pure
    calendar structure -- i.e. how much a model can get for free.
    """
    from statsmodels.tsa.seasonal import MSTL

    s = pd.Series(np.asarray(series, dtype=np.float64), index=timestamps)
    res = MSTL(s, periods=list(periods)).fit()

    seasonal = res.seasonal
    if isinstance(seasonal, pd.Series):
        seasonal = seasonal.to_frame(f"seasonal_{periods[0]}")

    components = {"trend": res.trend}
    for col in seasonal.columns:
        components[col] = seasonal[col]
    components["remainder"] = res.resid

    fig, axes = plt.subplots(len(components) + 1, 1,
                             figsize=(13, 1.8 * (len(components) + 1)), sharex=True)
    axes[0].plot(s.index, s.values, lw=0.5, color="#22405e")
    axes[0].set_ylabel("observed")
    for ax, (label, comp) in zip(axes[1:], components.items()):
        ax.plot(comp.index, comp.values, lw=0.6, color="#a54242")
        ax.set_ylabel(label)
    axes[-1].set_xlabel("date")
    fig.suptitle("MSTL decomposition (daily + weekly seasonality)", y=0.995)
    fig.tight_layout()

    total_var = float(np.var(s.values))
    stats = {f"var_share_{k}": float(np.var(v.values)) / total_var for k, v in components.items()}
    resid = res.resid.values
    z = (resid - np.nanmean(resid)) / np.nanstd(resid)
    stats["n_anomalies_|z|>4"] = int(np.sum(np.abs(z) > 4))
    stats["anomaly_timestamps"] = [str(t) for t in s.index[np.abs(z) > 4][:10]]
    return _save(fig, out_dir, name), stats


def plot_weekly_profile(series: np.ndarray, timestamps: pd.DatetimeIndex,
                        out_dir: Path, name: str = "fig05_profiles") -> Path:
    """Average day-of-week x hour heatmap plus mean daily profiles."""
    s = pd.Series(np.asarray(series, dtype=np.float64), index=timestamps)
    pivot = s.groupby([s.index.dayofweek, s.index.hour]).mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    im = axes[0].imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
    axes[0].set_yticks(range(7))
    axes[0].set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set_xlabel("hour of day")
    axes[0].set_title("(a) Mean activity by weekday and hour")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    for d, lbl in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        axes[1].plot(pivot.columns, pivot.loc[d], lw=1.2, label=lbl,
                     alpha=0.9 if d < 5 else 1.0,
                     ls="-" if d < 5 else "--")
    axes[1].set_xlabel("hour of day")
    axes[1].set_ylabel("mean activity")
    axes[1].set_title("(b) Mean daily profile by weekday")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_periodogram(series: np.ndarray, out_dir: Path,
                     name: str = "fig06_periodogram") -> tuple[Path, dict]:
    """Spectral confirmation of the 24 h and 7 d cycles."""
    from scipy.signal import periodogram

    s = np.asarray(series, dtype=np.float64)
    freqs, power = periodogram(s - s.mean(), fs=1.0)
    with np.errstate(divide="ignore"):
        period = 1.0 / freqs
    keep = (period > 2) & (period < 3000)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.semilogy(period[keep] / 6.0, power[keep], lw=0.7, color="#22405e")
    for hrs, lbl in ((24, "24 h"), (12, "12 h"), (168, "7 d")):
        ax.axvline(hrs, color="crimson", ls=":", lw=1)
        ax.text(hrs, power[keep].max(), lbl, rotation=90, fontsize=7, color="crimson",
                va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("period (hours, log scale)")
    ax.set_ylabel("spectral power")
    ax.set_title("Periodogram of Internet activity")
    fig.tight_layout()

    top_idx = np.argsort(power[keep])[::-1][:5]
    stats = {"dominant_periods_hours": [round(float(p) / 6.0, 2)
                                        for p in period[keep][top_idx]]}
    return _save(fig, out_dir, name), stats
