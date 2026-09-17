#!/usr/bin/env python
"""Stage 2 - exploratory and time-series analysis (Task 2).

Usage
-----
    python scripts/02_run_eda.py

Outputs figures fig01..fig06 and the tables that back the discussion:
    results/tables/top_squares.csv
    results/tables/distribution_stats.json
    results/tables/series_comparison.csv
    results/tables/stationarity.csv
    results/tables/timeseries_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import eda  # noqa: E402
from src.data.loader import get_series, load_processed  # noqa: E402
from src.utils.config import load_config, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-decomposition", action="store_true",
                    help="MSTL over 8,928 points takes a few minutes.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    figures, tables = cfg.path("figures_dir"), cfg.path("tables_dir")

    matrix, timestamps, square_ids, meta = load_processed(
        cfg.path("processed_dir"), cfg["dataset"]["activity"], mmap=True
    )
    print(f"Matrix {meta['shape']} ({meta['on_disk_mb']} MiB on disk), "
          f"{meta['start']} -> {meta['end']}")
    print(f"Missing cells filled: {meta['missing']['missing_fraction']:.4%} "
          f"(policy: {meta['missing']['policy']})")

    # ---- 2.1 spatial distribution -----------------------------------------
    print("\n=== 2.1 Spatial distribution ===")
    totals = eda.total_traffic_per_square(matrix)
    path, stats = eda.plot_spatial_distribution(totals, figures)
    with open(tables / "distribution_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"-> {path}")

    top3 = eda.top_k_squares(totals, square_ids, k=3)
    top_table = pd.DataFrame({
        "rank": range(1, 11),
        "square_id": eda.top_k_squares(totals, square_ids, k=10),
        "total_activity": np.sort(totals)[::-1][:10],
    })
    top_table.to_csv(tables / "top_squares.csv", index=False)
    print(f"\nTop 3 squares by total Internet traffic: {top3}")
    print(top_table.to_string(index=False))

    # ---- 2.2 selected series ----------------------------------------------
    print("\n=== 2.2 Time series, first two weeks ===")
    extra = [int(s) for s in cfg["eda"]["extra_squares"]]
    selected = top3 + extra
    start, end = cfg["eda"]["first_two_weeks_start"], cfg["eda"]["first_two_weeks_end"]
    print(f"-> {eda.plot_selected_series(matrix, square_ids, timestamps, selected, start, end, figures)}")

    comp = eda.compare_series_statistics(matrix, square_ids, timestamps, selected, start, end)
    comp.to_csv(tables / "series_comparison.csv")
    print(comp.to_string())

    # ---- 2.3 characterisation of the busiest square -----------------------
    target = top3[0]
    print(f"\n=== 2.3 Time-series analysis of square {target} ===")
    series = get_series(matrix, square_ids, target)
    ts_stats: dict = {"square_id": target}

    path, acf_stats = eda.plot_acf_pacf(series, figures, int(cfg["eda"]["acf_max_lag"]))
    ts_stats["acf"] = acf_stats
    print(f"-> {path}")
    print(json.dumps(acf_stats, indent=2))

    stat = eda.stationarity_tests(series, seasonal_period=cfg["dataset"]["intervals_per_day"])
    stat.to_csv(tables / "stationarity.csv")
    print("\nStationarity tests:")
    print(stat.to_string())

    print(f"-> {eda.plot_weekly_profile(series, timestamps, figures)}")
    path, spec = eda.plot_periodogram(series, figures)
    ts_stats["spectrum"] = spec
    print(f"-> {path}  dominant periods (h): {spec['dominant_periods_hours']}")

    if not args.skip_decomposition:
        print("\nRunning MSTL (this takes a few minutes)...")
        path, dec = eda.plot_decomposition(series, timestamps, figures)
        ts_stats["decomposition"] = dec
        print(f"-> {path}")
        print(json.dumps(dec, indent=2))

    with open(tables / "timeseries_stats.json", "w", encoding="utf-8") as fh:
        json.dump(ts_stats, fh, indent=2, default=str)

    with open(tables / "top3.json", "w", encoding="utf-8") as fh:
        json.dump({"top3": top3, "extra": extra}, fh, indent=2)
    print(f"\nSaved top-3 selection to {tables / 'top3.json'} "
          "(read by scripts/03_run_experiments.py)")


if __name__ == "__main__":
    main()
