"""Supervised-learning framing for one-step-ahead forecasting.

Two rules are enforced here because they are the classic silent mark-losers:

1. **Splits are strictly chronological.** No shuffling, ever. The validation
   week precedes the test week, which precedes nothing.
2. **The scaler is fitted on the training split only** and inverted before any
   metric is computed, so no test-set statistic ever leaks into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

class Scaler:
    """Optional ``log1p`` followed by standard or min-max scaling.

    ``log1p`` is applied because the marginal distribution of per-square
    traffic is strongly right-skewed and bursty; compressing it stabilises the
    gradient scale and stops rare spikes dominating the squared-error loss.
    It is invertible (``expm1``), so metrics are still reported in the original
    activity units.
    """

    def __init__(self, log1p: bool = True, method: str = "standard"):
        self.log1p = log1p
        self.method = method
        self.a_ = 0.0  # offset
        self.b_ = 1.0  # scale
        self.fitted_ = False

    def fit(self, x: np.ndarray) -> "Scaler":
        z = np.log1p(np.clip(x, 0, None)) if self.log1p else np.asarray(x, dtype=np.float64)
        if self.method == "standard":
            self.a_ = float(z.mean())
            self.b_ = float(z.std()) or 1.0
        elif self.method == "minmax":
            self.a_ = float(z.min())
            self.b_ = float(z.max() - z.min()) or 1.0
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        self.fitted_ = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Scaler.fit must be called on the training split first.")
        z = np.log1p(np.clip(x, 0, None)) if self.log1p else np.asarray(x, dtype=np.float64)
        return ((z - self.a_) / self.b_).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64) * self.b_ + self.a_
        if self.log1p:
            z = np.expm1(np.clip(z, None, 50.0))  # clip guards against overflow
        return np.clip(z, 0, None).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass
class SplitIndex:
    """Positional boundaries of the three chronological splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    timestamps: pd.DatetimeIndex

    def sizes(self) -> dict:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def make_splits(timestamps: pd.DatetimeIndex, splits_cfg: dict) -> SplitIndex:
    tz = timestamps.tz

    def _stamp(key: str) -> pd.Timestamp:
        ts = pd.Timestamp(splits_cfg[key])
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)

    def _mask(a: str, b: str) -> np.ndarray:
        return np.flatnonzero((timestamps >= _stamp(a)) & (timestamps <= _stamp(b)))

    return SplitIndex(
        train=_mask("train_start", "train_end"),
        val=_mask("val_start", "val_end"),
        test=_mask("test_start", "test_end"),
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

# Italian public holidays inside the observation window. 1 Nov 2013 is All
# Saints' Day and opens the dataset with an atypical, holiday-shaped day.
IT_HOLIDAYS = {"2013-11-01", "2013-12-08", "2013-12-25", "2013-12-26", "2014-01-01"}


def calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical time-of-day / day-of-week encodings plus weekend & holiday flags.

    Returns an ``(T, 6)`` float32 array. Sine/cosine pairs are used so that
    23:50 and 00:00 are adjacent in feature space rather than maximally distant.
    """
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    tod = 2 * np.pi * minute_of_day.to_numpy() / (24 * 60)
    dow = timestamps.dayofweek.to_numpy()
    dow_rad = 2 * np.pi * dow / 7
    is_weekend = (dow >= 5).astype(np.float32)
    is_holiday = np.isin(timestamps.strftime("%Y-%m-%d"), list(IT_HOLIDAYS)).astype(np.float32)
    return np.stack(
        [np.sin(tod), np.cos(tod), np.sin(dow_rad), np.cos(dow_rad), is_weekend, is_holiday],
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

@dataclass
class WindowedData:
    """Windows for one split.

    ``X`` has shape ``(N, L)``: the L values immediately preceding the target.
    ``y`` has shape ``(N,)``: the value at the target time, in scaled space.
    ``y_raw`` is the same target in original activity units.
    ``t_index`` gives the positional index of each target in the full series.
    """

    X: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    t_index: np.ndarray
    static: np.ndarray | None = None  # (N, F) calendar features at target time

    def __len__(self) -> int:
        return len(self.y)


def sliding_windows(series_scaled: np.ndarray, L: int, horizon: int = 1):
    """Build every valid (input window, target) pair.

    For target position ``t`` the input is ``series[t - L - horizon + 1 : t - horizon + 1]``.
    With ``horizon=1`` this is simply the L values up to and including ``t-1``.
    """
    n = len(series_scaled)
    view = np.lib.stride_tricks.sliding_window_view(series_scaled, L)  # (n-L+1, L)
    n_windows = n - L - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"Series of length {n} is too short for L={L}, horizon={horizon}.")
    X = np.ascontiguousarray(view[:n_windows])
    t_index = np.arange(L + horizon - 1, n)
    return X, t_index


def build_dataset(
    series_raw: np.ndarray,
    timestamps: pd.DatetimeIndex,
    split: SplitIndex,
    L: int,
    horizon: int = 1,
    scaler: Scaler | None = None,
    use_time_features: bool = True,
) -> tuple[dict[str, WindowedData], Scaler]:
    """Scale, window and split one square's series in a leak-free order.

    Order matters: fit the scaler on the *training timestamps only*, transform
    the whole series with those statistics, then window, then assign each
    window to a split by the position of its **target**. A window whose target
    lies in the test week may legitimately use history from the validation week
    -- that is exactly what an operational one-step-ahead forecaster sees.
    """
    series_raw = np.asarray(series_raw, dtype=np.float32)
    scaler = scaler or Scaler()
    if not scaler.fitted_:
        scaler.fit(series_raw[split.train])
    scaled = scaler.transform(series_raw)

    X, t_index = sliding_windows(scaled, L, horizon)
    y = scaled[t_index]
    y_raw = series_raw[t_index]

    static = None
    if use_time_features:
        static = calendar_features(timestamps)[t_index]

    out: dict[str, WindowedData] = {}
    for name, positions in (("train", split.train), ("val", split.val), ("test", split.test)):
        sel = np.flatnonzero(np.isin(t_index, positions))
        out[name] = WindowedData(
            X=X[sel],
            y=y[sel],
            y_raw=y_raw[sel],
            t_index=t_index[sel],
            static=None if static is None else static[sel],
        )
    return out, scaler
