#!/usr/bin/env python
"""Stage 1 - build the processed activity matrix and the memory evidence.

Usage
-----
    python scripts/01_build_dataset.py                 # full ingest + benchmark
    python scripts/01_build_dataset.py --benchmark-only
    python scripts/01_build_dataset.py --limit-days 5  # quick smoke run

Outputs
-------
    data/processed/internet_matrix.npy      (10000, 8928) float32
    data/processed/observed_mask.npy
    data/processed/timestamps.parquet, square_ids.npy, meta.json
    results/tables/memory_benchmark.csv     <- Table 1 of the report
    results/tables/ingest_profile.csv       <- per-file memory/time trace
    results/figures/fig00_memory.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import (  # noqa: E402
    benchmark_single_day,
    build_activity_matrix,
    discover_raw_files,
    save_processed,
)
from src.utils.config import load_config, set_seed  # noqa: E402
from src.utils.profiling import hardware_info  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit-days", type=int, default=None,
                    help="Ingest only the first N days (for a quick smoke test).")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ds = cfg["dataset"]

    raw_dir = Path(cfg.path("raw_dir"))
    tables = cfg.path("tables_dir")
    figures = cfg.path("figures_dir")

    files = discover_raw_files(raw_dir)
    print(f"Found {len(files)} daily files in {raw_dir}")
    print(json.dumps(hardware_info(), indent=2))

    # ---- Task 1 evidence: naive vs optimised on one day --------------------
    if not args.skip_benchmark:
        print("\n=== Memory benchmark on a single day ===")
        bench = benchmark_single_day(files[0], activity=ds["activity"],
                                     chunksize=int(ds["chunksize"]))
        bench.to_csv(tables / "memory_benchmark.csv", index=False)
        print(bench.to_string(index=False))
        naive, opt = bench.loc[0], bench.loc[1]
        print(f"\nDataFrame footprint: {naive['df_memory_mb']:.1f} MiB -> "
              f"{opt['df_memory_mb']:.1f} MiB "
              f"({bench.loc[1, 'reduction_vs_naive_pct']:.1f}% reduction)")
        print(f"Aggregates identical: {bench.loc[0, 'checksum_match']}")

        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.bar(["naive\n(all cols, default dtypes)", "optimised\n(3 cols, narrow, chunked)"],
               [naive["df_memory_mb"], opt["df_memory_mb"]],
               color=["#c1543a", "#4c9a52"], width=0.55)
        for i, v in enumerate([naive["df_memory_mb"], opt["df_memory_mb"]]):
            ax.text(i, v, f"{v:,.0f} MiB", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("peak in-memory DataFrame size (MiB)")
        ax.set_title(f"Memory footprint, one day ({files[0].name})", loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / "fig00_memory.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        if args.benchmark_only:
            return

    # ---- Full ingest -------------------------------------------------------
    print("\n=== Streaming ingest ===")
    subset = files[: args.limit_days] if args.limit_days else files
    result = build_activity_matrix(
        raw_dir=raw_dir,
        activity=ds["activity"],
        n_squares=int(ds["n_squares"]),
        intervals_per_day=int(ds["intervals_per_day"]),
        chunksize=int(ds["chunksize"]),
        timezone=ds["timezone"],
        files=subset,
    )

    profile = pd.DataFrame([r.as_row() for r in result.reports])
    profile.to_csv(tables / "ingest_profile.csv", index=False)

    meta = save_processed(result, cfg.path("processed_dir"), ds["activity"],
                          missing_policy=ds["missing_policy"])
    print("\n=== Saved ===")
    print(json.dumps(meta, indent=2))
    print(f"\nPeak per-file Python allocation: {profile['python_peak_mb'].max():.1f} MiB")
    print(f"Max process RSS during ingest:   {profile['rss_peak_mb'].max():.1f} MiB")
    print(f"Total ingest wall time:          {profile['wall_seconds'].sum():.1f} s")


if __name__ == "__main__":
    main()
