"""Naive baselines.

These are *not* among the three models the brief asks for -- they are the
reference every serious forecasting paper reports. For one-step-ahead
prediction on smooth 10-minute data, persistence is brutally strong. If a
neural network cannot beat it, that is a genuine finding and belongs in the
report rather than being quietly omitted.
"""

from __future__ import annotations

import numpy as np


class Persistence:
    r""":math:`\hat{x}(t+1) = x(t)` -- the random-walk forecast."""

    name = "Persistence"
    requires_training = False

    def fit(self, *args, **kwargs) -> "Persistence":
        return self

    def predict(self, series_raw: np.ndarray, t_index: np.ndarray) -> np.ndarray:
        return np.asarray(series_raw, dtype=np.float64)[t_index - 1]


class SeasonalNaive:
    r""":math:`\hat{x}(t+1) = x(t + 1 - m)` with ``m`` = one day by default."""

    requires_training = False

    def __init__(self, seasonal_period: int = 144):
        self.m = seasonal_period
        self.name = f"SeasonalNaive(m={seasonal_period})"

    def fit(self, *args, **kwargs) -> "SeasonalNaive":
        return self

    def predict(self, series_raw: np.ndarray, t_index: np.ndarray) -> np.ndarray:
        return np.asarray(series_raw, dtype=np.float64)[t_index - self.m]


class DriftedSeasonalNaive:
    """Seasonal naive corrected by the recent level shift.

    Adds the average difference between the last ``k`` observations and their
    values one season earlier. Cheap, and often surprisingly competitive.
    """

    requires_training = False

    def __init__(self, seasonal_period: int = 144, window: int = 6):
        self.m = seasonal_period
        self.window = window
        self.name = f"DriftedSeasonalNaive(m={seasonal_period})"

    def fit(self, *args, **kwargs) -> "DriftedSeasonalNaive":
        return self

    def predict(self, series_raw: np.ndarray, t_index: np.ndarray) -> np.ndarray:
        s = np.asarray(series_raw, dtype=np.float64)
        base = s[t_index - self.m]
        drift = np.zeros_like(base)
        for k in range(1, self.window + 1):
            drift += s[t_index - k] - s[t_index - k - self.m]
        return np.clip(base + drift / self.window, 0, None)


def default_baselines(seasonal_period: int = 144) -> list:
    return [Persistence(), SeasonalNaive(seasonal_period), DriftedSeasonalNaive(seasonal_period)]
