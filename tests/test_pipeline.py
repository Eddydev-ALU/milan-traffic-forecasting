"""Correctness tests for the parts that fail silently if they are wrong.

Window alignment and scaler leakage do not raise exceptions when broken -- they
just quietly produce optimistic numbers. These tests pin both down.

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windows import (  # noqa: E402
    Scaler,
    build_dataset,
    calendar_features,
    make_splits,
    sliding_windows,
)
from src.evaluate import mae, mase, rmse, wape  # noqa: E402
from src.models.baselines import Persistence, SeasonalNaive  # noqa: E402


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    series = (100 + 40 * np.sin(2 * np.pi * t / 144) + rng.normal(0, 3, n)).astype(np.float32)
    series = np.clip(series, 0, None)
    ts = pd.date_range("2013-11-01", periods=n, freq="10min", tz="Europe/Rome")
    return series, ts


SPLITS = {
    "train_start": "2013-11-01 00:00", "train_end": "2013-11-08 23:50",
    "val_start": "2013-11-09 00:00", "val_end": "2013-11-10 23:50",
    "test_start": "2013-11-11 00:00", "test_end": "2013-11-13 23:50",
}


def test_sliding_window_alignment():
    """X[i] must be exactly the L values immediately before target t_index[i]."""
    y = np.arange(50, dtype=np.float32)
    L = 5
    X, t_index = sliding_windows(y, L, horizon=1)
    assert t_index[0] == L
    assert len(X) == len(t_index) == len(y) - L
    for i in (0, 7, len(X) - 1):
        t = t_index[i]
        np.testing.assert_array_equal(X[i], y[t - L: t])
        assert X[i][-1] == y[t - 1]  # last input is the previous observation


def test_horizon_two_skips_a_step():
    y = np.arange(30, dtype=np.float32)
    X, t_index = sliding_windows(y, 4, horizon=2)
    t = t_index[0]
    np.testing.assert_array_equal(X[0], y[t - 5: t - 1])


def test_scaler_roundtrip_and_train_only_fit():
    rng = np.random.default_rng(1)
    train = rng.gamma(2.0, 50.0, 500)
    test = rng.gamma(2.0, 50.0, 200) * 10  # very different scale
    sc = Scaler(log1p=True, method="standard").fit(train)
    a, b = sc.a_, sc.b_
    sc.transform(test)
    assert (sc.a_, sc.b_) == (a, b), "transform must not refit on new data"
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(train)), train, rtol=1e-4)


def test_splits_are_chronological_and_disjoint():
    _, ts = _toy()
    split = make_splits(ts, SPLITS)
    assert split.train.max() < split.val.min() < split.test.min()
    assert set(split.train).isdisjoint(split.val)
    assert set(split.val).isdisjoint(split.test)


def test_build_dataset_targets_match_raw_series():
    series, ts = _toy()
    split = make_splits(ts, SPLITS)
    data, scaler = build_dataset(series, ts, split, L=24, horizon=1)
    for name in ("train", "val", "test"):
        d = data[name]
        np.testing.assert_allclose(d.y_raw, series[d.t_index], rtol=1e-5)
        # Scaled targets must invert back to the raw targets.
        np.testing.assert_allclose(scaler.inverse_transform(d.y), d.y_raw, rtol=1e-3)


def test_no_target_leakage_across_splits():
    series, ts = _toy()
    split = make_splits(ts, SPLITS)
    data, _ = build_dataset(series, ts, split, L=24)
    assert set(data["train"].t_index).isdisjoint(data["test"].t_index)
    assert data["train"].t_index.max() < data["test"].t_index.min()


def test_scaler_statistics_use_training_data_only():
    series, ts = _toy()
    split = make_splits(ts, SPLITS)
    _, sc_all = build_dataset(series, ts, split, L=24)
    expected = Scaler(log1p=True, method="standard").fit(series[split.train])
    assert np.isclose(sc_all.a_, expected.a_) and np.isclose(sc_all.b_, expected.b_)


def test_persistence_matches_definition():
    series, ts = _toy()
    split = make_splits(ts, SPLITS)
    data, _ = build_dataset(series, ts, split, L=24)
    t_idx = data["test"].t_index
    np.testing.assert_array_equal(Persistence().predict(series, t_idx), series[t_idx - 1])
    np.testing.assert_array_equal(
        SeasonalNaive(144).predict(series, t_idx), series[t_idx - 144]
    )


def test_metrics_against_hand_computed_values():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([12.0, 18.0, 33.0, 36.0])
    # absolute errors: 2, 2, 3, 4  ->  sum 11, mean 2.75
    assert np.isclose(mae(y, p), 2.75)
    assert np.isclose(rmse(y, p), np.sqrt((4 + 4 + 9 + 16) / 4))
    assert np.isclose(wape(y, p), 100 * 11 / 100)
    assert np.isclose(mae(y, y), 0.0)


def test_mase_is_one_for_seasonal_naive_on_its_own_scale():
    rng = np.random.default_rng(3)
    train = rng.normal(100, 10, 1000)
    y = rng.normal(100, 10, 200)
    perfect = y.copy()
    assert np.isclose(mase(y, perfect, train, seasonal_period=144), 0.0)


def test_calendar_features_are_cyclical():
    ts = pd.date_range("2013-11-01", periods=288, freq="10min", tz="Europe/Rome")
    f = calendar_features(ts)
    assert f.shape == (288, 6)
    # midnight on consecutive days must have identical time-of-day encoding
    np.testing.assert_allclose(f[0, :2], f[144, :2], atol=1e-6)
    assert f[:, 5].max() == 1.0  # 1 Nov 2013 is flagged as a holiday


def test_fourier_terms_are_periodic():
    from src.models.sarimax_fourier import fourier_terms

    t = np.arange(500)
    F = fourier_terms(t, [{"period": 144, "n_terms": 3}])
    assert F.shape == (500, 6)
    np.testing.assert_allclose(F[0], F[144], atol=1e-10)


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", __file__]))
