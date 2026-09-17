#!/usr/bin/env python
"""Stage 4 - failure analysis (rubric: 'Failure Analysis & Critical Reflection').

Reloads the saved predictions and answers three questions with numbers rather
than adjectives:

1. *When* does each model fail -- by hour, by weekday, over the test week?
2. Are the predictions simply lagging the signal (the persistence trap)?
3. Which individual timestamps are worst, and what happened there?

Usage
-----
    python scripts/04_failure_analysis.py
    python scripts/04_failure_analysis.py --square 5161
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import plots  # noqa: E402
from src.data.loader import get_series, load_processed  # noqa: E402
from src.data.windows import make_splits  # noqa: E402
from src.evaluate import error_profile, lag_diagnostic  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--square", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    figures, tables = cfg.path("figures_dir"), cfg.path("tables_dir")
    pred_dir = Path(cfg.path("predictions_dir"))

    matrix, timestamps, square_ids, _ = load_processed(
        cfg.path("processed_dir"), cfg["dataset"]["activity"], mmap=True
    )
    split = make_splits(timestamps, cfg["splits"])

    files = sorted(pred_dir.glob("pred_sq*.npy"))
    if not files:
        raise SystemExit("No predictions found. Run scripts/03_run_experiments.py first.")

    squares = sorted({int(f.stem.split("_")[1][2:]) for f in files})
    target = args.square or squares[0]
    print(f"Areas with saved predictions: {squares}; analysing {target}")

    L = int(cfg["forecasting"]["sequence_length"])
    horizon = int(cfg["forecasting"]["horizon"])
    series = get_series(matrix, square_ids, target)
    t_index = split.test[split.test >= L + horizon - 1]
    y_true = series[t_index]
    ts = timestamps[t_index]

    profiles, summary = {}, {}
    for f in files:
        if int(f.stem.split("_")[1][2:]) != target:
            continue
        model = f.stem.split("_", 2)[2]
        pred = np.load(f)
        if len(pred) != len(y_true):
            print(f"  [skip] {model}: length {len(pred)} != {len(y_true)}")
            continue
        prof = error_profile(y_true, pred, ts)
        profiles[model] = prof
        lag = lag_diagnostic(y_true, pred)
        worst = prof.nlargest(5, "abs_error")[["timestamp", "actual", "predicted", "abs_error"]]
        summary[model] = {
            "MAE": float(prof["abs_error"].mean()),
            "MAE_night_00_06": float(prof.loc[prof["hour"] < 6, "abs_error"].mean()),
            "MAE_day_08_20": float(
                prof.loc[(prof["hour"] >= 8) & (prof["hour"] < 20), "abs_error"].mean()),
            "MAE_weekday": float(prof.loc[~prof["is_weekend"], "abs_error"].mean()),
            "MAE_weekend": float(prof.loc[prof["is_weekend"], "abs_error"].mean()),
            "bias_mean_error": float(prof["error"].mean()),
            "best_backshift": lag["best_backshift"],
            "lag_verdict": lag["verdict"],
            "mae_by_backshift": {str(k): round(v, 2) for k, v in lag["mae_by_backshift"].items()},
            "worst_timestamps": worst.assign(
                timestamp=worst["timestamp"].astype(str)).to_dict("records"),
        }
        print(f"\n--- {model} ---")
        print(json.dumps({k: v for k, v in summary[model].items()
                          if k != "worst_timestamps"}, indent=2, default=str))
        print("worst 5 timestamps:")
        print(worst.to_string(index=False))

    if profiles:
        print(f"\n-> {plots.plot_error_breakdown(profiles, target, figures)}")

    # Zoomed overlay on the single worst hour across models -- the concrete
    # "period for which one or more models perform poorly" the brief requires.
    worst_hour = (pd.concat(profiles.values())
                  .groupby("timestamp")["abs_error"].mean().idxmax())
    lo = (pd.Timestamp(worst_hour) - pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
    hi = (pd.Timestamp(worst_hour) + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
    preds = {m: p["predicted"].to_numpy() for m, p in profiles.items()}
    print(f"\nWorst period across models: {worst_hour}")
    print(f"-> {plots.plot_model_overlay(ts, y_true, preds, target, figures, zoom=(lo, hi))}")

    with open(tables / f"failure_analysis_sq{target}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    pd.DataFrame(summary).T.drop(columns=["worst_timestamps", "mae_by_backshift"]).to_csv(
        tables / f"failure_summary_sq{target}.csv"
    )
    print(f"\nSaved -> {tables / f'failure_analysis_sq{target}.json'}")


if __name__ == "__main__":
    main()
