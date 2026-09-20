# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

One-step-ahead forecasting of Internet activity in the Telecom Italia Milan
grid (10,000 squares, 10-minute resolution, Nov 2013 – Jan 2014), comparing a
statistical model, a recurrent network and a convolutional network.

**Research question.** How do different sequential models compare for one-step-ahead
mobile network traffic forecasting, and how does their performance vary across
geographical areas with different traffic characteristics?

---

## Headline results

I evaluated three models plus three naive baselines on a held-out test week
(16–22 December 2013) across the three busiest squares. No single model wins
everywhere, and the pattern tracks the burstiness I measured during the
exploratory analysis:

| Area | Character | Best model | MAE | WAPE |
|---|---|---|---|---|
| 5161 (Duomo) | bursty, CV 0.885 | TCN | 78.66 | 5.44% |
| 5059 (Duomo/Via Spadari) | smoother, CV 0.716 | SARIMAX(1,1,1)+Fourier | 67.22 | 5.31% |
| 5259 (Brera/Cordusio) | bursty, CV 0.879 | TCN | 60.30 | 4.77% |

All three fitted models beat every naive baseline in every area. The TCN wins
the two burstiest areas; the linear statistical model wins the one comparatively
smooth area. Full numbers are in `results/tables/metrics_all.csv`.

## Quick start

```bash
git clone https://github.com/Eddydev-ALU/milan-traffic-forecasting.git
cd milan-traffic-forecasting

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# On a CPU-only machine, install the far smaller CPU build of PyTorch instead:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

make test     # 12 unit tests, ~3 s
make smoke    # whole pipeline on synthetic data, ~2 min, no download required
```

`make smoke` generates fake files in the dataset's exact on-disk format and runs
all four stages against them, so the environment can be verified in two minutes
without downloading 20 GB.

## Getting the real data

The daily files come from the Harvard Dataverse mirrors for the Telecom Italia
Big Data Challenge dataset and belong in `data/raw/`:

```
data/raw/
├── sms-call-internet-mi-2013-11-01.txt
├── sms-call-internet-mi-2013-11-02.txt
└── ... (62 files)
```

Each file is tab-separated with no header:

| # | column | notes |
|---|--------|-------|
| 1 | `square_id` | 1–10,000 |
| 2 | `time_interval` | epoch **milliseconds**, 10-minute slots |
| 3 | `country_code` | the reason the files are so long |
| 4–7 | `smsin`, `smsout`, `callin`, `callout` | often empty |
| 8 | `internet` | the forecasting target |

Section 4.2 of the report also identifies squares by location, which needs the
Milano Grid geojson in `data/grid/`.

## Reproducing the results

```bash
make data          # ingest + memory benchmark      ~3 min
make eda           # figures 1-6 and the EDA tables ~10 min (MSTL dominates)
make tune          # validation-week grid search    ~40 min on CPU
make experiments   # final runs, three areas        ~5 min
make failure       # failure analysis               ~1 min
```

Each stage writes to `results/` and is independently re-runnable. Stages 2–4
read the processed matrix, so `make data` only needs to run once.

Useful flags:

```bash
python scripts/01_build_dataset.py --benchmark-only   # just the memory table
python scripts/01_build_dataset.py --limit-days 5     # partial ingest
python scripts/03_run_experiments.py --squares 5161   # one area
python scripts/03_run_experiments.py --no-baselines
python scripts/04_failure_analysis.py --square 5161
python scripts/05_identify_squares.py                 # square -> lat/lon + map link
```

## Repository layout

```
configs/
  config.yaml              all paths, splits and hyperparameters
  config_synth.yaml        same code path, synthetic-data sizes
src/
  utils/config.py          config loading, path resolution, seeding
  utils/profiling.py       memory instrumentation and the timing harness
  data/loader.py           chunked ingest -> (10000, 8928) float32 matrix
  data/windows.py          scaling, chronological splits, sliding windows
  analysis/eda.py          spatial distribution, ACF/PACF, MSTL, stationarity
  analysis/plots.py        prediction, error and training-curve figures
  models/baselines.py      persistence, seasonal naive, drifted seasonal naive
  models/sarimax_fourier.py  Model 1
  models/neural.py         Models 2 and 3 (LSTM, TCN)
  evaluate.py              MAE / RMSE / MAPE / WAPE / MASE, lag diagnostic
  experiment.py            per-area orchestration
  tuning.py                grid search with a persistent experiment log
scripts/                   00-05, the runnable stages
tests/                     unit tests
results/figures, tables, predictions
```

## Design decisions worth knowing about

**Memory.** The raw files are long rather than wide: every `(square, slot)` pair
is split into one row per interacting country code. Aggregating over
`country_code` *on arrival*, while reading only three columns with narrow
dtypes in chunks, collapses millions of rows into a fixed 10,000 × 144 block
per day. The full result is 10,000 × 8,928 float32 ≈ 341 MiB, which fits in RAM
and memory-maps in milliseconds. `results/tables/memory_benchmark.csv` records
the naive-versus-optimised comparison (peak DataFrame 295.6 → 26.7 MiB, a 91.0%
reduction), with a checksum confirming the two paths produce identical
aggregates. The full 62-file ingest peaks at 1,631.3 MiB RSS and takes 136.8 s.

**Why SARIMAX with Fourier terms, not SARIMA.** Seasonal ARIMA with `s=144`
needs a state vector of at least 144 and is computationally infeasible in
`statsmodels`; `s=1008` is hopeless; and it cannot represent two seasonalities
at once. The standard remedy is to move seasonality into exogenous Fourier
regressors and keep a low-order non-seasonal ARMA for the residual dependence.

**Three baselines, not two.** Alongside persistence and seasonal naive I added a
drifted seasonal naive, which corrects the seasonal-naive forecast by the recent
level shift. Plain seasonal naive collapses badly at square 5259 (MAE 470 against
60 for the TCN) because it has no mechanism to notice a weekday/weekend boundary,
and the drifted variant separates that failure from the underlying idea.

**Leakage control.** Splits are strictly chronological and never shuffled. The
scaler is fitted on the training split alone and inverted before any metric is
computed. `tests/test_pipeline.py` asserts both, because neither failure raises
an exception — they just quietly produce optimistic numbers.

**MAPE is reported twice.** Night-time traffic approaches zero, so unrestricted
MAPE is dominated by a handful of 4 a.m. observations. Both the unrestricted
value and one restricted to points above the 10th percentile are reported, and
WAPE and MASE are included because they are scale-free without the zero-denominator
pathology.

**Timing.** `src/utils/profiling.py:timeit` discards warm-up runs, calls
`torch.cuda.synchronize()` around GPU work so timings aren't optimistic, and
reports mean ± std over repeats. Baselines and SARIMAX use three repeats after
one discarded warm-up; neural training uses three full independent fits.
`results/tables/hardware.json` records the exact machine.

## Hyperparameter tuning

`results/tables/experiment_log.csv` is the full record: 56 trials (4 SARIMAX
orders, 36 LSTM configurations, 16 TCN configurations), each with its validation
MAE, cost, and a `rationale` column explaining what that trial contributes.
Tuning ran on square 5161's validation week only; the winning settings are
written back into `configs/config.yaml` and reused for the other areas.

Selected configurations:

| Model | Configuration | Val MAE |
|---|---|---|
| SARIMAX | order (1,1,1); 5 daily + 3 weekly Fourier harmonics | 101.41 |
| LSTM | L=24, hidden 128, 1 layer, lr 0.003 | 103.27 |
| TCN | L=144, kernel 5, dropout 0.1, lr 0.003 | 97.75 |

Two findings from the sweep are worth calling out. A second LSTM layer lost to
its one-layer twin at every width and every window length. And the TCN's kernel
size matters more than its input window, because four dilated layers give a
receptive field of 1+2(k−1)(2⁴−1) — 61 steps at k=3 but 121 steps at k=5 — so a
288-step window is history the model cannot actually reach.

## Hardware, and the unused Colab notebook

Everything reported here ran on one machine: an Apple Silicon MacBook Pro,
8 cores, 16 GB RAM, Python 3.14.6, PyTorch 2.14.0, **CPU only** — no GPU and no
MPS backend. The complete neural grid search took about 40 minutes of CPU time,
so a GPU was never necessary.

`notebooks/colab_train.ipynb` is a working Colab setup for running the tuning
stage on a T4 if the grid is ever expanded. I did not use it, and none of the
results or timings in this repository came from it.

## Configuration

Everything tunable lives in `configs/config.yaml` — nothing is hard-coded in the
scripts. To change the input window length, edit `forecasting.sequence_length`
(or the per-model `sequence_length`); to change the evaluation week, edit
`splits`.

## Testing

```bash
pytest tests/ -q
```

Covers window alignment (`X[i]` really is the L values before target `i`),
scaler round-trip and train-only fitting, split disjointness, baseline
definitions, hand-computed metric values, and Fourier-term periodicity.

## Known limitations

- Traffic is modelled one area at a time; spatial correlation between
  neighbouring squares is not exploited.
- Country-level structure is discarded during aggregation.
- All three modelled areas are central, high-traffic squares, so the findings
  should not be assumed to hold for low-traffic peripheral areas.
- Results come from a single random seed and a single test week.

## Report and video

- Report: `report/` (submitted separately as PDF)
- Video: link in the report's References section

## Data citation

G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan
and the Province of Trentino," *Scientific Data*, vol. 2, 150055, 2015.
https://doi.org/10.1038/sdata.2015.55
