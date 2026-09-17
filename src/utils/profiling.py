"""Memory and timing instrumentation.

Task 1 of the brief asks for *evidence* of memory usage before and after
optimisation, and Task 4-IV asks for exact training/inference timings together
with the method used to obtain them. Both are produced here so that every
number quoted in the report is reproducible by re-running one function.
"""

from __future__ import annotations

import gc
import platform
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import psutil

_PROC = psutil.Process()


def rss_mb() -> float:
    """Resident set size of this process in MiB."""
    return _PROC.memory_info().rss / 1024**2


@dataclass
class MemoryReport:
    label: str
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_peak_mb: float = 0.0
    python_peak_mb: float = 0.0
    wall_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def rss_delta_mb(self) -> float:
        return self.rss_after_mb - self.rss_before_mb

    def as_row(self) -> dict:
        row = {
            "label": self.label,
            "rss_before_mb": round(self.rss_before_mb, 1),
            "rss_after_mb": round(self.rss_after_mb, 1),
            "rss_peak_mb": round(self.rss_peak_mb, 1),
            "rss_delta_mb": round(self.rss_delta_mb, 1),
            "python_peak_mb": round(self.python_peak_mb, 1),
            "wall_seconds": round(self.wall_seconds, 2),
        }
        row.update(self.extra)
        return row


@contextmanager
def measure_memory(label: str) -> Iterator[MemoryReport]:
    """Measure RSS growth, Python-level peak allocation and wall time.

    ``tracemalloc`` captures peak *Python object* allocation, which is the
    number that reveals transient spikes (e.g. an intermediate DataFrame that
    is freed before the block ends). RSS captures what the OS actually had to
    reserve, which is what limits you on a laptop.
    """
    gc.collect()
    report = MemoryReport(label=label, rss_before_mb=rss_mb())
    tracemalloc.start()
    start = time.perf_counter()
    try:
        yield report
    finally:
        report.wall_seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report.python_peak_mb = peak / 1024**2
        report.rss_after_mb = rss_mb()
        report.rss_peak_mb = max(report.rss_after_mb, report.rss_before_mb + report.python_peak_mb)


def dataframe_memory_mb(df, deep: bool = True) -> float:
    """Exact in-memory footprint of a DataFrame, including object strings."""
    return float(df.memory_usage(deep=deep).sum()) / 1024**2


def array_memory_mb(arr: np.ndarray) -> float:
    return arr.nbytes / 1024**2


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def _sync() -> None:
    """Block until queued GPU work has finished, so timings are not optimistic."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timeit(fn: Callable[[], Any], n_repeats: int = 3, warmup: int = 1) -> dict:
    """Time ``fn`` over repeats, discarding warm-up runs.

    Returns mean/std/min in seconds plus the raw samples, so the report can
    state exactly how the statistic was computed.
    """
    samples: list[float] = []
    result = None
    for i in range(warmup + n_repeats):
        gc.collect()
        _sync()
        t0 = time.perf_counter()
        result = fn()
        _sync()
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            samples.append(elapsed)
    arr = np.asarray(samples)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min_s": float(arr.min()),
        "n_repeats": n_repeats,
        "warmup": warmup,
        "samples_s": samples,
        "result": result,
    }


def hardware_info() -> dict:
    """Machine description to quote verbatim in the report (Task 4-IV)."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "device": "cpu",
        "gpu": "none",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
            info["cuda"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"unavailable ({type(exc).__name__})"
    return info
