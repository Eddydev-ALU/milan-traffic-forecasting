"""Memory-efficient ingest of the Telecom Italia Milan CDR dataset.

Raw layout
----------
One tab-separated file per day, named ``sms-call-internet-mi-YYYY-MM-DD.txt``::

    square_id  time_interval(ms)  country_code  smsin  smsout  callin  callout  internet

The file is long rather than wide because every ``(square_id, time_interval)``
slot is split into one row per interacting ``country_code``. That is the single
biggest source of bloat: a day holds ~300k logical slots (10,000 squares x 144
intervals) but several million physical rows.

Optimisation strategy
---------------------
1. Stream one day at a time -- the 62 files are never co-resident.
2. Read only the three columns we need (``usecols``), with explicit narrow
   dtypes, so the parser never materialises the country column or float64
   copies of columns we discard.
3. Aggregate over ``country_code`` on arrival via ``np.add.at``, collapsing
   millions of rows into a fixed 10,000 x 144 float32 block.
4. Accumulate into a single pre-allocated ``float32`` matrix. The end state is
   10,000 x 8,928 x 4 B = 341 MiB for the whole city and two months.
5. Persist as ``.npy``; every later stage memory-maps it in milliseconds.

Trade-offs (discuss these in the report)
----------------------------------------
* Summing over ``country_code`` discards country-level structure. It is the
  right call for this research question (total per-area traffic) but it is a
  loss of information, not a free lunch.
* ``float32`` keeps ~7 significant digits. Activity values are already a
  de-identified proxy measure, so this is far below the noise floor -- but it
  is a deliberate precision/space trade.
* Chunked parsing trades CPU time for RAM: more Python-level iterations, lower
  peak memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.profiling import dataframe_memory_mb, measure_memory

RAW_COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet",
]

ACTIVITY_COLUMNS = ("smsin", "smsout", "callin", "callout", "internet")

# Narrow dtypes for the columns we keep. square_id max is 10,000 -> int16 would
# overflow at 32,767? No: 10,000 < 32,767, so int16 is safe and halves the cost
# versus int32 (and is 4x cheaper than the default int64).
OPTIMISED_DTYPES = {
    "square_id": "int16",
    "time_interval": "int64",  # epoch ms needs the full width
    "internet": "float32",
    "smsin": "float32",
    "smsout": "float32",
    "callin": "float32",
    "callout": "float32",
}

MS_PER_SLOT = 600_000  # 10 minutes
FILENAME_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_raw_files(raw_dir: str | Path) -> list[Path]:
    """Return the daily files sorted by the date embedded in their name."""
    raw_dir = Path(raw_dir)
    files = sorted(
        (p for p in raw_dir.rglob("*") if p.is_file() and p.suffix in {".txt", ".csv"}
         and FILENAME_DATE.search(p.name)),
        key=lambda p: FILENAME_DATE.search(p.name).group(0),
    )
    if not files:
        raise FileNotFoundError(
            f"No daily files found under {raw_dir}. Expected names like "
            "'sms-call-internet-mi-2013-11-01.txt'."
        )
    return files


def file_date(path: Path) -> pd.Timestamp:
    m = FILENAME_DATE.search(path.name)
    if m is None:  # pragma: no cover - guarded by discover_raw_files
        raise ValueError(f"Cannot parse a date from {path.name}")
    return pd.Timestamp(m.group(0))


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    matrix: np.ndarray            # (n_squares, T) float32
    observed: np.ndarray          # (n_squares, T) bool -- True where a raw row existed
    timestamps: pd.DatetimeIndex  # length T, tz-aware
    square_ids: np.ndarray        # (n_squares,) int32, 1..n_squares
    reports: list                 # per-file MemoryReport objects


def build_activity_matrix(
    raw_dir: str | Path,
    activity: str = "internet",
    n_squares: int = 10_000,
    intervals_per_day: int = 144,
    chunksize: int = 2_000_000,
    timezone: str = "Europe/Rome",
    files: Iterable[Path] | None = None,
    verbose: bool = True,
) -> IngestResult:
    """Stream the raw files into one pre-allocated ``float32`` matrix."""
    files = list(files) if files is not None else discover_raw_files(raw_dir)
    dates = [file_date(p) for p in files]
    n_days = len(files)

    # Global time grid. The observation window (Nov 2013 - Jan 2014) contains no
    # DST transition, so a fixed local->UTC offset is safe; we still go through
    # tz-aware timestamps so the code stays correct if the window is changed.
    start_local = dates[0].tz_localize(timezone)
    timestamps = pd.date_range(
        start=start_local, periods=n_days * intervals_per_day, freq="10min", tz=timezone
    )
    start_ms = int(start_local.tz_convert("UTC").value // 1_000_000)
    n_slots = len(timestamps)

    matrix = np.zeros((n_squares, n_slots), dtype=np.float32)
    observed = np.zeros((n_squares, n_slots), dtype=bool)
    reports = []

    usecols = ["square_id", "time_interval", activity]
    dtypes = {c: OPTIMISED_DTYPES[c] for c in usecols}

    for i, path in enumerate(files):
        with measure_memory(f"ingest:{path.name}") as rep:
            rows = 0
            reader = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=RAW_COLUMNS,
                usecols=usecols,
                dtype=dtypes,
                chunksize=chunksize,
                engine="c",
            )
            for chunk in reader:
                rows += len(chunk)
                _accumulate_chunk(chunk, matrix, observed, activity, start_ms, n_slots, n_squares)
                del chunk
            rep.extra = {"raw_rows": rows, "file": path.name}
        reports.append(rep)
        if verbose:
            print(
                f"[{i + 1:>2}/{n_days}] {path.name}: {rep.extra['raw_rows']:>9,} rows "
                f"-> {rep.wall_seconds:5.1f}s, peak python {rep.python_peak_mb:6.1f} MiB, "
                f"RSS {rep.rss_after_mb:7.1f} MiB",
                flush=True,
            )

    square_ids = np.arange(1, n_squares + 1, dtype=np.int32)
    return IngestResult(matrix, observed, timestamps, square_ids, reports)


def _accumulate_chunk(
    chunk: pd.DataFrame,
    matrix: np.ndarray,
    observed: np.ndarray,
    activity: str,
    start_ms: int,
    n_slots: int,
    n_squares: int,
) -> None:
    """Fold one chunk into the matrix, summing over country codes.

    ``np.add.at`` is unbuffered, so repeated (row, col) pairs -- which is
    exactly what the country split produces -- accumulate correctly, including
    across chunk boundaries.
    """
    values = chunk[activity].to_numpy(dtype=np.float32, copy=False)
    valid = ~np.isnan(values)
    if not valid.any():
        return

    rows = chunk["square_id"].to_numpy(dtype=np.int32, copy=False)[valid] - 1
    cols = ((chunk["time_interval"].to_numpy(dtype=np.int64, copy=False)[valid] - start_ms)
            // MS_PER_SLOT).astype(np.int64)
    values = values[valid]

    in_range = (rows >= 0) & (rows < n_squares) & (cols >= 0) & (cols < n_slots)
    if not in_range.all():
        rows, cols, values = rows[in_range], cols[in_range], values[in_range]

    np.add.at(matrix, (rows, cols), values)
    observed[rows, cols] = True


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------

def fill_missing(
    matrix: np.ndarray, observed: np.ndarray, policy: str = "interpolate"
) -> tuple[np.ndarray, dict]:
    """Resolve slots for which no raw row existed.

    An absent row means "no CDR was recorded", which is ambiguous: it can be a
    genuine zero (a rural square at 04:00) or a gap in the feed. For the dense
    squares we forecast, a gap in the middle of a busy day is almost certainly
    missing data, so linear interpolation along time is the defensible default.
    The policy is configurable and the statistics are reported so the choice is
    auditable.
    """
    stats = {
        "n_cells": int(matrix.size),
        "n_missing": int((~observed).sum()),
        "missing_fraction": float((~observed).mean()),
        "policy": policy,
    }
    if policy == "zero" or stats["n_missing"] == 0:
        return matrix, stats
    if policy == "nan":
        out = matrix.copy()
        out[~observed] = np.nan
        return out, stats

    out = matrix.astype(np.float32, copy=True)
    idx = np.arange(matrix.shape[1])
    rows_with_gaps = np.flatnonzero((~observed).any(axis=1))
    for r in rows_with_gaps:
        mask = observed[r]
        if mask.sum() < 2:
            out[r] = 0.0
            continue
        out[r] = np.interp(idx, idx[mask], out[r][mask]).astype(np.float32)
    return out, stats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_processed(
    result: IngestResult,
    out_dir: str | Path,
    activity: str,
    missing_policy: str = "interpolate",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filled, miss_stats = fill_missing(result.matrix, result.observed, missing_policy)

    np.save(out_dir / f"{activity}_matrix.npy", filled)
    np.save(out_dir / "observed_mask.npy", result.observed)
    np.save(out_dir / "square_ids.npy", result.square_ids)
    # Stored as int64 nanoseconds + a tz string: no parquet engine required,
    # and the round-trip is exact.
    np.save(out_dir / "timestamps_ns.npy",
            result.timestamps.tz_convert("UTC").to_numpy("datetime64[ns]").astype(np.int64))

    meta = {
        "activity": activity,
        "shape": list(filled.shape),
        "dtype": str(filled.dtype),
        "start": str(result.timestamps[0]),
        "end": str(result.timestamps[-1]),
        "freq": "10min",
        "timezone": str(result.timestamps.tz),
        "on_disk_mb": round(filled.nbytes / 1024**2, 1),
        "missing": miss_stats,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def load_processed(processed_dir: str | Path, activity: str = "internet", mmap: bool = True):
    """Load the matrix (memory-mapped by default) plus its time index."""
    processed_dir = Path(processed_dir)
    matrix = np.load(
        processed_dir / f"{activity}_matrix.npy", mmap_mode="r" if mmap else None
    )
    square_ids = np.load(processed_dir / "square_ids.npy")
    with open(processed_dir / "meta.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    ns = np.load(processed_dir / "timestamps_ns.npy")
    timestamps = pd.DatetimeIndex(pd.to_datetime(ns, unit="ns", utc=True)).tz_convert(
        meta.get("timezone", "Europe/Rome")
    )
    return matrix, timestamps, square_ids, meta


def get_series(matrix: np.ndarray, square_ids: np.ndarray, square_id: int) -> np.ndarray:
    """Extract one square's full time series as a contiguous float32 array."""
    pos = int(np.flatnonzero(square_ids == square_id)[0])
    return np.ascontiguousarray(matrix[pos], dtype=np.float32)


# ---------------------------------------------------------------------------
# Before/after benchmark for the report
# ---------------------------------------------------------------------------

def benchmark_single_day(path: str | Path, activity: str = "internet",
                         chunksize: int = 2_000_000) -> pd.DataFrame:
    """Compare a naive full-file read against the optimised streaming read.

    This produces the "before and after" evidence Task 1 asks for. Run it on
    ONE day file; extrapolate to 62 days in the report and state that you did.
    """
    path = Path(path)
    rows = []

    # --- (A) Naive: default dtypes, all columns, single read -----------------
    with measure_memory("naive_full_read") as rep:
        df = pd.read_csv(path, sep="\t", header=None, names=RAW_COLUMNS)
        rep.extra = {
            "df_memory_mb": round(dataframe_memory_mb(df), 1),
            "rows": len(df),
            "columns": df.shape[1],
            "strategy": "read_csv(default dtypes, all 8 columns)",
        }
        naive_total = df.groupby(["square_id", "time_interval"])[activity].sum()
        rep.extra["unique_slots"] = len(naive_total)
        naive_checksum = float(np.nansum(naive_total.to_numpy()))
    del df, naive_total
    rows.append(rep.as_row())

    # --- (B) Optimised: 3 columns, narrow dtypes, chunked + aggregate --------
    with measure_memory("optimised_chunked") as rep:
        acc: dict[tuple[int, int], float] = {}
        total_rows = 0
        reader = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=RAW_COLUMNS,
            usecols=["square_id", "time_interval", activity],
            dtype={k: OPTIMISED_DTYPES[k] for k in ("square_id", "time_interval", activity)},
            chunksize=chunksize,
        )
        peak_chunk_mb = 0.0
        for chunk in reader:
            total_rows += len(chunk)
            peak_chunk_mb = max(peak_chunk_mb, dataframe_memory_mb(chunk))
            g = chunk.groupby(["square_id", "time_interval"], sort=False)[activity].sum()
            for key, val in g.items():
                if not np.isnan(val):
                    acc[key] = acc.get(key, 0.0) + float(val)
            del chunk, g
        rep.extra = {
            "df_memory_mb": round(peak_chunk_mb, 1),
            "rows": total_rows,
            "columns": 3,
            "unique_slots": len(acc),
            "strategy": f"usecols=3, narrow dtypes, chunksize={chunksize:,}, aggregate on arrival",
        }
        opt_checksum = float(sum(acc.values()))
    del acc
    rows.append(rep.as_row())

    out = pd.DataFrame(rows)
    out["checksum_match"] = bool(np.isclose(naive_checksum, opt_checksum, rtol=1e-4))
    naive_mb = out.loc[0, "df_memory_mb"]
    opt_mb = out.loc[1, "df_memory_mb"]
    out["reduction_vs_naive_pct"] = [0.0, round(100 * (1 - opt_mb / naive_mb), 1)]
    return out
