"""Model 1: SARIMAX with Fourier seasonal regressors.

Why not plain SARIMA?
---------------------
The data has a daily period of 144 and a weekly period of 1008. A seasonal
ARIMA with ``s=144`` requires a state vector of length >= 144, so estimation is
computationally infeasible in ``statsmodels``, and ``s=1008`` is hopeless.
It also cannot represent two seasonalities at once.

The standard remedy (Hyndman & Athanasopoulos, *FPP3*, ch. 12) is to move the
seasonality into **exogenous Fourier terms** and keep a low-order,
non-seasonal ARMA for the remaining short-range dependence::

    x_t = beta' F_t + eta_t,     eta_t ~ ARMA(p, q)

``F_t`` holds sin/cos pairs at the daily and weekly fundamental frequencies and
their harmonics. This gives smooth multi-seasonal structure with a handful of
parameters and a tiny state vector, which is why this is the standard remedy
for multi-seasonal series with long periods.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False


def fourier_terms(t: np.ndarray, specs: list[dict]) -> np.ndarray:
    """Build the Fourier design matrix.

    Parameters
    ----------
    t : integer positions in the global time grid (0, 1, 2, ... ).
    specs : e.g. ``[{"period": 144, "n_terms": 5}, {"period": 1008, "n_terms": 3}]``.

    Returns an ``(len(t), 2 * sum(n_terms))`` float64 array.
    """
    t = np.asarray(t, dtype=np.float64)
    cols = []
    for spec in specs:
        period = float(spec["period"])
        for k in range(1, int(spec["n_terms"]) + 1):
            ang = 2.0 * np.pi * k * t / period
            cols.append(np.sin(ang))
            cols.append(np.cos(ang))
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


class SarimaxFourier:
    """One-step-ahead SARIMAX forecaster with multi-seasonal Fourier regressors."""

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 0, 1),
        fourier_specs: list[dict] | None = None,
        log1p: bool = True,
        trend: str = "c",
        maxiter: int = 200,
    ):
        if not HAS_STATSMODELS:  # pragma: no cover
            raise ImportError("statsmodels is required: pip install statsmodels")
        self.order = tuple(order)
        self.fourier_specs = fourier_specs or [
            {"period": 144, "n_terms": 5},
            {"period": 1008, "n_terms": 3},
        ]
        self.log1p = log1p
        self.trend = trend
        self.maxiter = maxiter
        self.name = f"SARIMAX{self.order}+Fourier"
        self.res_ = None
        self.fit_start_ = None
        self.fit_end_ = None

    # -- transforms ---------------------------------------------------------
    def _fwd(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.log1p(np.clip(x, 0, None)) if self.log1p else x

    def _inv(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return np.clip(np.expm1(np.clip(z, None, 50.0)), 0, None) if self.log1p else np.clip(z, 0, None)

    # -- fit / predict ------------------------------------------------------
    def fit(self, series_raw: np.ndarray, train_pos: np.ndarray) -> "SarimaxFourier":
        train_pos = np.asarray(train_pos)
        if not np.all(np.diff(train_pos) == 1):
            raise ValueError("SARIMAX requires a contiguous training block.")
        self.fit_start_, self.fit_end_ = int(train_pos[0]), int(train_pos[-1])

        endog = self._fwd(np.asarray(series_raw)[train_pos])
        exog = fourier_terms(train_pos, self.fourier_specs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.res_ = model.fit(disp=False, maxiter=self.maxiter)
        return self

    def predict(self, series_raw: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Genuine one-step-ahead predictions at the given positions.

        The fitted parameters are **frozen** (``refit=False``); only the Kalman
        filter state is advanced through the newly observed data. This mirrors
        deployment: coefficients estimated once offline, state updated as each
        new 10-minute observation arrives.
        """
        if self.res_ is None:
            raise RuntimeError("Call fit() first.")
        target_pos = np.asarray(target_pos)
        if not np.all(np.diff(target_pos) == 1):
            raise ValueError("Target positions must be contiguous.")

        series = np.asarray(series_raw)
        start, end = int(target_pos[0]), int(target_pos[-1])
        if start <= self.fit_end_:
            raise ValueError("Target block must start after the training block.")

        extend = np.arange(self.fit_end_ + 1, end + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res2 = self.res_.append(
                self._fwd(series[extend]),
                exog=fourier_terms(extend, self.fourier_specs),
                refit=False,
            )
        fitted = np.asarray(res2.fittedvalues, dtype=np.float64)
        # fittedvalues are indexed from fit_start_; convert absolute -> local.
        local = target_pos - self.fit_start_
        return self._inv(fitted[local])

    # -- diagnostics --------------------------------------------------------
    @property
    def aic(self) -> float:
        return float(self.res_.aic) if self.res_ is not None else float("nan")

    @property
    def bic(self) -> float:
        return float(self.res_.bic) if self.res_ is not None else float("nan")

    def summary(self):  # pragma: no cover - interactive use
        return self.res_.summary()

    def residual_diagnostics(self) -> dict:
        """Ljung-Box on the residuals: is any structure left unmodelled?"""
        from statsmodels.stats.diagnostic import acorr_ljungbox

        resid = np.asarray(self.res_.resid, dtype=np.float64)
        lb = acorr_ljungbox(resid, lags=[10, 144], return_df=True)
        return {
            "resid_std": float(resid.std()),
            "ljung_box": lb.to_dict(),
            "aic": self.aic,
            "bic": self.bic,
        }


def grid_search_order(
    series_raw: np.ndarray,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    orders: list,
    fourier_specs: list[dict],
    log1p: bool = True,
    metric_fn=None,
) -> list[dict]:
    """Select ``(p, d, q)`` by validation MAE, with AIC recorded alongside.

    Validation error is the primary criterion because it matches how the model
    is actually judged; AIC is logged so the report can note where the two
    criteria agree or disagree.
    """
    from ..evaluate import mae as _mae

    metric_fn = metric_fn or _mae
    y_val = np.asarray(series_raw)[val_pos]
    log = []
    for order in orders:
        try:
            model = SarimaxFourier(order=order, fourier_specs=fourier_specs, log1p=log1p)
            model.fit(series_raw, train_pos)
            pred = model.predict(series_raw, val_pos)
            log.append(
                {
                    "order": tuple(order),
                    "val_MAE": float(metric_fn(y_val, pred)),
                    "aic": model.aic,
                    "bic": model.bic,
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001 - record and continue the sweep
            log.append({"order": tuple(order), "val_MAE": float("inf"),
                        "aic": float("nan"), "bic": float("nan"),
                        "status": f"failed: {type(exc).__name__}"})
    return sorted(log, key=lambda r: r["val_MAE"])
