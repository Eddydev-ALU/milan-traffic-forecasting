"""Hyperparameter search with a persistent experiment log.

Each experiment is documented with its parameters, its performance and the
reasoning behind it. ``ExperimentLog`` writes one row per experiment to CSV so
the table in the report is generated rather than retyped, and the ``rationale``
column records what that configuration was testing and what it showed.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data.windows import Scaler, build_dataset
from .evaluate import mae


class ExperimentLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            self.rows = pd.read_csv(self.path).to_dict("records")

    def add(self, **kwargs) -> None:
        kwargs.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.rows.append(kwargs)
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def already_done(self, **key) -> bool:
        """True if a trial with exactly these key fields is already logged.

        Colab sessions disconnect. Because every trial is flushed to CSV as it
        finishes, re-running the same command after a disconnect resumes from
        where it stopped instead of repeating hours of work.
        """
        for row in self.rows:
            if all(str(row.get(k)) == str(v) for k, v in key.items()):
                status = str(row.get("status", "ok"))
                if not status.startswith("failed"):
                    return True
        return False


def _grid(space: dict):
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def tune_neural(
    arch: str,
    series_raw: np.ndarray,
    timestamps,
    split,
    cfg,
    log: ExperimentLog,
    square_id: int,
    max_trials: int | None = None,
    verbose: bool = True,
) -> dict:
    """Grid-search an LSTM or TCN on the **validation** week only.

    The test week is never touched here. Selection is by validation MAE in
    original activity units, which is the same quantity used for early stopping
    and for the final results tables.
    """
    from .models.neural import NeuralForecaster

    space = dict(cfg["tuning"][arch])
    base = dict(cfg["models"][arch])
    fc = cfg["forecasting"]
    combos = list(_grid(space))
    if max_trials:
        combos = combos[:max_trials]

    best, best_mae = None, float("inf")
    for i, params in enumerate(combos, 1):
        L = int(params.pop("sequence_length", fc["sequence_length"]))
        merged = {**base, **params}

        key = {"square_id": square_id, "model": arch.upper(), "sequence_length": L,
               **{k: str(v) for k, v in params.items()}}
        if log.already_done(**key):
            if verbose:
                print(f"  [{i}/{len(combos)}] L={L} {params} -- already logged, skipping")
            continue
        data, scaler = build_dataset(
            series_raw, timestamps, split, L=L, horizon=int(fc["horizon"]),
            scaler=Scaler(log1p=fc["scaler"]["log1p"], method=fc["scaler"]["method"]),
            use_time_features=fc["use_time_features"],
        )
        arch_kwargs = {k: v for k, v in merged.items()
                       if k not in {"lr", "batch_size", "max_epochs", "patience",
                                    "grad_clip", "loss"}}
        if arch == "tcn":
            arch_kwargs["channels"] = tuple(arch_kwargs["channels"])

        t0 = time.perf_counter()
        model = NeuralForecaster(
            arch=arch, scaler=scaler, lr=float(merged["lr"]),
            batch_size=int(merged["batch_size"]), max_epochs=int(merged["max_epochs"]),
            patience=int(merged["patience"]), grad_clip=float(merged["grad_clip"]),
            loss=merged["loss"], seed=int(cfg["seed"]), **arch_kwargs,
        ).fit(data["train"], data["val"])
        elapsed = time.perf_counter() - t0

        val_mae = mae(data["val"].y_raw, model.predict(data["val"]))
        improved = val_mae < best_mae
        log.add(
            experiment=len(log.rows) + 1, square_id=square_id, model=arch.upper(),
            sequence_length=L, **{k: str(v) for k, v in params.items()},
            val_MAE=round(val_mae, 2), train_seconds=round(elapsed, 1),
            epochs_run=model.history.epochs_run, best_epoch=model.history.best_epoch,
            stopped_early=model.history.stopped_early,
            n_parameters=model.n_parameters(),
            rationale="",  # annotated after the sweep, in the log itself
        )
        if verbose:
            flag = "  <-- best so far" if improved else ""
            print(f"  [{i}/{len(combos)}] L={L} {params} val_MAE={val_mae:,.1f}{flag}")
        if improved:
            best_mae = val_mae
            best = {"sequence_length": L, **merged, "val_MAE": val_mae}

    if best is None:
        # Everything was skipped (a fully resumed run): recover the winner from
        # the log so the caller still gets an answer.
        frame = log.to_frame()
        if not frame.empty and "val_MAE" in frame:
            frame = frame[(frame["model"] == arch.upper())
                          & (frame["square_id"].astype(str) == str(square_id))]
            if not frame.empty:
                row = frame.loc[frame["val_MAE"].astype(float).idxmin()]
                best = row.to_dict()
                best_mae = float(row["val_MAE"])

    return {"best": best, "best_val_mae": best_mae, "n_trials": len(combos)}


def tune_sarimax(series_raw, split, cfg, log: ExperimentLog, square_id: int,
                 verbose: bool = True) -> dict:
    """Order selection for the SARIMAX model, logged in the same table."""
    from .models.sarimax_fourier import grid_search_order

    sc = cfg["models"]["sarimax"]
    results = grid_search_order(
        series_raw, split.train, split.val,
        orders=[tuple(o) for o in cfg["tuning"]["sarimax"]["order"]],
        fourier_specs=[dict(f) for f in sc["fourier"]],
        log1p=bool(sc["log1p"]),
    )
    for r in results:
        if log.already_done(square_id=square_id, model="SARIMAX", order=str(r["order"])):
            continue
        log.add(experiment=len(log.rows) + 1, square_id=square_id, model="SARIMAX",
                order=str(r["order"]), val_MAE=round(r["val_MAE"], 2),
                aic=round(r["aic"], 1) if np.isfinite(r["aic"]) else None,
                bic=round(r["bic"], 1) if np.isfinite(r["bic"]) else None,
                status=r["status"], rationale="")
        if verbose:
            print(f"  order={r['order']}  val_MAE={r['val_MAE']:,.1f}  AIC={r['aic']:,.0f}")
    return {"best": results[0] if results else None, "all": results}
