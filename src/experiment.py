"""Orchestration: run every model on every area, with timing and logging.

One function, ``run_area``, owns the whole per-area pipeline so that the three
models are guaranteed to see identical splits, identical scaling and identical
evaluation code. Anything that differs between models is a modelling choice,
not an accident of the harness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data.windows import Scaler, SplitIndex, build_dataset
from .evaluate import all_metrics, error_profile, lag_diagnostic
from .models.baselines import default_baselines
from .models.sarimax_fourier import SarimaxFourier
from .utils.profiling import hardware_info, timeit


@dataclass
class ModelRun:
    model: str
    square_id: int
    metrics: dict
    train_time: dict
    infer_time: dict
    predictions: np.ndarray = field(repr=False, default=None)
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        out = {"model": self.model, "square_id": self.square_id}
        out.update({k: v for k, v in self.metrics.items() if k != "mape_threshold"})
        out["train_time_s"] = round(self.train_time.get("mean_s", 0.0), 3)
        out["train_time_std_s"] = round(self.train_time.get("std_s", 0.0), 3)
        out["infer_time_s"] = round(self.infer_time.get("mean_s", 0.0), 4)
        out["infer_time_per_step_ms"] = self.extra.get("infer_per_step_ms")
        out["n_parameters"] = self.extra.get("n_parameters")
        return out


def run_area(
    series_raw: np.ndarray,
    timestamps: pd.DatetimeIndex,
    split: SplitIndex,
    square_id: int,
    cfg,
    include_baselines: bool = True,
    verbose: bool = True,
) -> tuple[list[ModelRun], dict]:
    """Train, time and evaluate all models for a single geographical area."""
    fc = cfg["forecasting"]
    mc = cfg["models"]
    tc = cfg["timing"]
    L = int(fc["sequence_length"])
    horizon = int(fc["horizon"])
    n_repeats, warmup = int(tc["n_repeats"]), int(tc["warmup"])

    data, scaler = build_dataset(
        series_raw, timestamps, split, L=L, horizon=horizon,
        scaler=Scaler(log1p=fc["scaler"]["log1p"], method=fc["scaler"]["method"]),
        use_time_features=fc["use_time_features"],
    )
    test = data["test"]
    y_test = test.y_raw
    test_ts = timestamps[test.t_index]
    train_series = series_raw[split.train]
    m = int(mc["baselines"]["seasonal_period"])

    runs: list[ModelRun] = []
    histories: dict = {}

    def _record(name, preds, train_time, infer_time, extra=None):
        met = all_metrics(y_test, preds, train_series=train_series, seasonal_period=m)
        extra = extra or {}
        extra["infer_per_step_ms"] = round(1000 * infer_time["mean_s"] / max(len(preds), 1), 4)
        runs.append(ModelRun(name, square_id, met, train_time, infer_time, preds, extra))
        if verbose:
            print(f"    {name:<28} MAE={met['MAE']:>10,.1f}  RMSE={met['RMSE']:>10,.1f} "
                  f" WAPE={met['WAPE_%']:>5.2f}%  train={train_time['mean_s']:.2f}s")

    # -- baselines ----------------------------------------------------------
    if include_baselines:
        for bl in default_baselines(m):
            t = timeit(lambda b=bl: b.predict(series_raw, test.t_index),
                       n_repeats=n_repeats, warmup=warmup)
            _record(bl.name, np.asarray(t["result"]), {"mean_s": 0.0, "std_s": 0.0}, t)

    # -- Model 1: SARIMAX + Fourier ----------------------------------------
    sc = mc["sarimax"]
    sarimax = SarimaxFourier(
        order=tuple(sc["order"]),
        fourier_specs=[dict(f) for f in sc["fourier"]],
        log1p=bool(sc["log1p"]),
        trend=sc["trend"],
        maxiter=int(sc["maxiter"]),
    )
    t_train = timeit(lambda: sarimax.fit(series_raw, split.train),
                     n_repeats=n_repeats, warmup=warmup)
    t_infer = timeit(lambda: sarimax.predict(series_raw, test.t_index),
                     n_repeats=n_repeats, warmup=warmup)
    _record(sarimax.name, np.asarray(t_infer["result"]), t_train, t_infer,
            {"aic": sarimax.aic, "bic": sarimax.bic,
             "n_parameters": int(len(sarimax.res_.params))})

    context = {
        "histories": histories,
        "hardware": hardware_info(),
        "test_timestamps": test_ts,
        "y_test": y_test,
        "scaler": scaler,
        "sequence_length": L,
    }

    # -- Models 2 & 3: LSTM and TCN ----------------------------------------
    try:
        from .models.neural import HAS_TORCH, NeuralForecaster

        if not HAS_TORCH:
            raise ImportError("torch unavailable")
    except Exception:
        print("    [warn] PyTorch unavailable - skipping LSTM and TCN.")
        return runs, context

    for arch, key in (("lstm", "lstm"), ("tcn", "tcn")):
        params = dict(mc[key])
        arch_kwargs = {k: v for k, v in params.items()
                       if k not in {"lr", "batch_size", "max_epochs", "patience",
                                    "grad_clip", "loss", "sequence_length"}}
        if arch == "tcn":
            arch_kwargs["channels"] = tuple(arch_kwargs["channels"])

        # Each architecture may have its own tuned window length. Rebuilding
        # per model is cheap and safe: every target in the test split lies far
        # past position max(L), so the set of forecasted timestamps -- and
        # hence y_test / test_ts from the shared `context` above -- is
        # identical regardless of which L built this model's windows.
        model_L = int(params.get("sequence_length", L))
        if model_L != L:
            model_data, model_scaler = build_dataset(
                series_raw, timestamps, split, L=model_L, horizon=horizon,
                scaler=Scaler(log1p=fc["scaler"]["log1p"], method=fc["scaler"]["method"]),
                use_time_features=fc["use_time_features"],
            )
        else:
            model_data, model_scaler = data, scaler

        def _make(_scaler=model_scaler):
            return NeuralForecaster(
                arch=arch, scaler=_scaler, lr=float(params["lr"]),
                batch_size=int(params["batch_size"]), max_epochs=int(params["max_epochs"]),
                patience=int(params["patience"]), grad_clip=float(params["grad_clip"]),
                loss=params["loss"], seed=int(cfg["seed"]), **arch_kwargs,
            )

        # Training is timed with a *single* repeat by default because a full
        # fit is expensive; set timing.n_repeats > 1 in the config for a proper
        # mean +- std and state in the report which you used.
        holder = {}

        def _fit(_train=model_data["train"], _val=model_data["val"]):
            model = _make()
            model.fit(_train, _val)
            holder["model"] = model
            return model

        t_train = timeit(_fit, n_repeats=max(n_repeats, 1), warmup=0)
        model = holder["model"]
        t_infer = timeit(lambda: model.predict(model_data["test"]), n_repeats=n_repeats, warmup=warmup)
        histories[model.name] = model.history
        _record(model.name, np.asarray(t_infer["result"]), t_train, t_infer,
                {"n_parameters": model.n_parameters(),
                 "receptive_field": model.receptive_field(),
                 "sequence_length": model_L,
                 "best_epoch": model.history.best_epoch,
                 "epochs_run": model.history.epochs_run,
                 "stopped_early": model.history.stopped_early})

    return runs, context


# ---------------------------------------------------------------------------
# Persistence of results
# ---------------------------------------------------------------------------

def save_runs(runs: list[ModelRun], context: dict, out_dirs: dict) -> pd.DataFrame:
    table = pd.DataFrame([r.row() for r in runs])
    pred_dir = Path(out_dirs["predictions"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        safe = r.model.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
        np.save(pred_dir / f"pred_sq{r.square_id}_{safe}.npy", r.predictions)
    if "test_timestamps" in context:
        pd.Series(context["test_timestamps"], name="timestamp").to_csv(
            pred_dir / "test_timestamps.csv", index=False
        )
    with open(Path(out_dirs["tables"]) / "hardware.json", "w", encoding="utf-8") as fh:
        json.dump(context["hardware"], fh, indent=2)
    return table


def failure_analysis(runs: list[ModelRun], context: dict, square_id: int) -> dict:
    """Assemble the evidence for the failure-analysis section."""
    ts = context["test_timestamps"]
    y_true = context["y_test"]
    profiles, diagnostics = {}, {}
    for r in runs:
        if r.square_id != square_id:
            continue
        prof = error_profile(y_true, r.predictions, ts)
        profiles[r.model] = prof
        worst = prof.nlargest(5, "abs_error")[["timestamp", "actual", "predicted", "abs_error"]]
        diagnostics[r.model] = {
            "lag": lag_diagnostic(y_true, r.predictions),
            "worst_timestamps": worst.assign(
                timestamp=worst["timestamp"].astype(str)
            ).to_dict("records"),
            "mae_night_00_06": float(prof.loc[prof["hour"] < 6, "abs_error"].mean()),
            "mae_day_08_20": float(
                prof.loc[(prof["hour"] >= 8) & (prof["hour"] < 20), "abs_error"].mean()),
            "mae_weekday": float(prof.loc[~prof["is_weekend"], "abs_error"].mean()),
            "mae_weekend": float(prof.loc[prof["is_weekend"], "abs_error"].mean()),
        }
    return {"profiles": profiles, "diagnostics": diagnostics}