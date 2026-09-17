#!/usr/bin/env python
"""Generate synthetic raw files in the exact Telecom Italia format.

Purpose: verify the whole pipeline end-to-end without waiting on the 20 GB
download, and give a reviewer a way to reproduce the mechanics of the project
in under a minute. The synthetic series deliberately contains a daily cycle, a
weekly cycle, a weekend effect and injected spikes, so the EDA and the models
have something real to find.

Usage
-----
    python scripts/00_make_synthetic_data.py --days 21 --squares 60 \
        --out data/raw_synth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--squares", type=int, default=60)
    ap.add_argument("--start", default="2013-11-01")
    ap.add_argument("--countries", type=int, default=3,
                    help="Rows per (square, slot) -- mimics the country-code split.")
    ap.add_argument("--out", default="data/raw_synth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    n_sq, per_day = args.squares, 144
    # Per-square scale is log-normal, reproducing the heavy right skew of the
    # real spatial distribution.
    scale = np.exp(rng.normal(3.0, 1.1, size=n_sq))
    phase = rng.uniform(-0.4, 0.4, size=n_sq)

    start = pd.Timestamp(args.start, tz="Europe/Rome")
    for d in range(args.days):
        day = start + pd.Timedelta(days=d)
        slots = pd.date_range(day, periods=per_day, freq="10min", tz="Europe/Rome")
        tod = 2 * np.pi * (slots.hour * 60 + slots.minute).to_numpy() / 1440.0
        weekend = 0.65 if day.dayofweek >= 5 else 1.0

        rows = []
        for s in range(n_sq):
            daily = 1.0 + 0.85 * np.sin(tod - 1.9 + phase[s]) + 0.35 * np.sin(2 * tod + 0.6)
            base = scale[s] * weekend * np.clip(daily, 0.06, None)
            noise = rng.normal(1.0, 0.09, size=per_day)
            values = np.clip(base * noise, 0, None)
            if rng.random() < 0.02:  # occasional event spike
                i = rng.integers(0, per_day)
                values[i: i + 3] *= rng.uniform(2.5, 6.0)
            # Split each slot across country codes so the ingest has real
            # aggregation work to do.
            parts = rng.dirichlet(np.ones(args.countries), size=per_day)
            for c in range(args.countries):
                rows.append(pd.DataFrame({
                    "square_id": s + 1,
                    "time_interval": slots.tz_convert("UTC").to_numpy("datetime64[ms]").astype("int64"),
                    "country_code": c + 1,
                    "smsin": np.nan,
                    "smsout": np.nan,
                    "callin": np.nan,
                    "callout": np.nan,
                    "internet": values * parts[:, c],
                }))
        df = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=d)
        path = out / f"sms-call-internet-mi-{day.strftime('%Y-%m-%d')}.txt"
        df.to_csv(path, sep="\t", header=False, index=False, na_rep="")
        print(f"wrote {path.name}  ({len(df):,} rows)")

    print(f"\nNow run:\n  python scripts/01_build_dataset.py "
          f"--config configs/config_synth.yaml")


if __name__ == "__main__":
    main()
