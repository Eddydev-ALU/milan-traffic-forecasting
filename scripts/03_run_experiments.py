#!/usr/bin/env python
"""Stage 3 - hyperparameter tuning and the forecasting experiments.

Usage
-----
    python scripts/03_run_experiments.py --tune          # validation-week search
    python scripts/03_run_experiments.py                 # final runs, 3 areas
    python scripts/03_run_experiments.py --squares 5161  # single area

Outputs
-------
    results/tables/experiment_log.csv        <- the iterative tuning table
    results/tables/metrics_sq<ID>.csv        <- one table per area
    results/tables/metrics_all.csv
    results/tables/timing.csv                <- train/infer timings
    results/figures/fig_pred_sq<ID>_<MODEL>.png   <- per-model prediction plots
    results/figures/fig_overlay_sq<ID>.png
    results/predictions/*.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analysis import eda, plots  # noqa: E402
from src.data.loader import get_series, load_processed  # noqa: E402
from src.data.windows import make_splits  # noqa: E402
from src.experiment import run_area, save_runs  # noqa: E402
from src.tuning import ExperimentLog, tune_neural, tune_sarimax  # noqa: E402
from src.utils.config import load_config, set_seed  # noqa: E402
from src.utils.profiling import hardware_info  # noqa: E402


def resolve_squares(cfg, matrix, square_ids, explicit=None) -> list[int]:
    if explicit:
        return [int(s) for s in explicit]
    cache = Path(cfg.path("tables_dir")) / "top3.json"
    if cache.exists():
        return json.loads(cache.read_text())["top3"]
    print("top3.json not found - recomputing totals (run 02_run_eda.py first to cache).")
    return eda.top_k_squares(eda.total_traffic_per_square(matrix), square_ids, k=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--squares", nargs="*", default=None)
    ap.add_argument("--tune", action="store_true", help="Run the validation-week grid search.")
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    figures, tables = cfg.path("figures_dir"), cfg.path("tables_dir")
    out_dirs = {"tables": tables, "figures": figures,
                "predictions": cfg.path("predictions_dir")}

    matrix, timestamps, square_ids, _ = load_processed(
        cfg.path("processed_dir"), cfg["dataset"]["activity"], mmap=True
    )
    split = make_splits(timestamps, cfg["splits"])
    print("Split sizes (10-min steps):", split.sizes())
    print("Test window:", timestamps[split.test[0]], "->", timestamps[split.test[-1]])
    print(json.dumps(hardware_info(), indent=2))

    squares = resolve_squares(cfg, matrix, square_ids, args.squares)
    print(f"Areas under study: {squares}")

    # ---- Optional tuning, on the busiest area only -------------------------
    if args.tune:
        log = ExperimentLog(tables / "experiment_log.csv")
        target = squares[0]
        series = get_series(matrix, square_ids, target)
        print(f"\n=== Tuning on square {target} (validation week only) ===")

        print("\n-- SARIMAX order search --")
        tune_sarimax(series, split, cfg, log, target)

        for arch in ("lstm", "tcn"):
            print(f"\n-- {arch.upper()} grid search --")
            try:
                best = tune_neural(arch, series, timestamps, split, cfg, log, target,
                                   max_trials=args.max_trials)
            except ImportError as exc:
                print(f"  [skip] {exc}")
                continue
            print(f"Best {arch.upper()}: {best['best']}")
            print("Update configs/config.yaml with these values, then re-run without --tune.")

        print(f"\nExperiment log -> {log.path}")
        return

    # ---- Final experiments -------------------------------------------------
    all_rows, all_context = [], {}
    for sid in squares:
        print(f"\n=== Square {sid} ===")
        series = get_series(matrix, square_ids, sid)
        runs, context = run_area(series, timestamps, split, sid, cfg,
                                 include_baselines=not args.no_baselines)
        table = save_runs(runs, context, out_dirs)
        table.to_csv(tables / f"metrics_sq{sid}.csv", index=False)
        all_rows.append(table)
        all_context[sid] = (runs, context)

        preds = {}
        for r in runs:
            met = r.metrics
            plots.plot_actual_vs_predicted(context["test_timestamps"], context["y_test"],
                                           r.predictions, r.model, sid, figures, met)
            preds[r.model] = r.predictions
        plots.plot_model_overlay(context["test_timestamps"], context["y_test"],
                                 preds, sid, figures)
        plots.plot_model_overlay(context["test_timestamps"], context["y_test"], preds, sid,
                                 figures, zoom=("2013-12-16 05:00", "2013-12-16 13:00"))
        if context.get("histories"):
            plots.plot_training_curves(context["histories"], figures)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(tables / "metrics_all.csv", index=False)

    timing = combined[["model", "square_id", "train_time_s", "train_time_std_s",
                       "infer_time_s", "infer_time_per_step_ms", "n_parameters"]]
    timing.to_csv(tables / "timing.csv", index=False)

    for metric in ("MAE", "RMSE", "WAPE_%"):
        plots.plot_metric_comparison(combined, figures, metric,
                                     name=f"fig_compare_{metric.replace('%', 'pct')}")

    print("\n=== Combined results ===")
    print(combined.to_string(index=False))
    print(f"\nTables -> {tables}\nFigures -> {figures}")


if __name__ == "__main__":
    main()
