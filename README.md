# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

One-step-ahead forecasting of Internet activity in the Telecom Italia Milan
grid (10,000 squares, 10-minute resolution, Nov 2013 – Jan 2014), comparing a
statistical model, a recurrent network and a convolutional network.

**Research question.** How do different sequential models compare for one-step-ahead
mobile network traffic forecasting, and how does their performance vary across
geographical areas with different traffic characteristics?

---

## Quick start

```bash
git clone https://github.com/<your-username>/milan-traffic-forecasting.git
cd milan-traffic-forecasting

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# On a CPU-only machine, install the far smaller CPU build of PyTorch instead:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

make test     # 12 unit tests, ~2 s
make smoke    # whole pipeline on synthetic data, ~2 min, no download required
```

`make smoke` generates fake files in the dataset's exact on-disk format and runs
all four stages against them. Run it before downloading 20 GB — it verifies the
environment in two minutes.

## Getting the real data

Download the daily files from the Harvard Dataverse mirrors listed in the
assignment brief and place them in `data/raw/`:

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

## Reproducing the results

```bash
make data          # ingest + memory benchmark      ~15-25 min
make eda           # figures 1-6 and the EDA tables ~10 min (MSTL dominates)
make tune          # validation-week grid search    ~1-3 h (GPU: much less)
make experiments   # final runs, three areas        ~30-90 min
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
```

## Running on Google Colab

Colab's only advantage for this project is the **GPU**: its free tier gives
~12.7 GB RAM and ephemeral disk, which is likely less than your own machine,
and free Drive is 15 GB — smaller than the raw dataset. So split the work:

| Stage | Where | Why |
|---|---|---|
| `make data` (ingest) | **Laptop** | Needs the full raw dataset; disk- and parser-bound, no GPU benefit |
| `make eda` | **Laptop** | NumPy/statsmodels, CPU-only |
| SARIMAX | **Laptop** | `statsmodels` is CPU-only; a GPU does nothing |
| LSTM / TCN tuning | **Colab GPU** | The only genuinely GPU-bound stage |
| `make failure` | Either | Seconds either way |

Upload only `data/processed/` to Drive — `internet_matrix.npy` (~341 MB) plus
three small metadata files. Then open `notebooks/colab_train.ipynb`, which
mounts Drive, clones the repo, writes a Colab config pointing results back at
Drive, and runs the grid search.

The tuning loop is **resume-safe**: every trial is flushed to
`experiment_log.csv` as it finishes and completed trials are skipped on the next
run, so a Colab disconnect costs you at most one trial rather than the session.

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
  models/baselines.py      persistence and seasonal naive
  models/sarimax_fourier.py  Model 1
  models/neural.py         Models 2 and 3 (LSTM, TCN)
  evaluate.py              MAE / RMSE / MAPE / WAPE / MASE, lag diagnostic
  experiment.py            per-area orchestration
  tuning.py                grid search with a persistent experiment log
scripts/                   00-04, the runnable stages
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
the naive-versus-optimised comparison, with a checksum confirming the two paths
produce identical aggregates.

**Why SARIMAX with Fourier terms, not SARIMA.** Seasonal ARIMA with `s=144`
needs a state vector of at least 144 and is computationally infeasible in
`statsmodels`; `s=1008` is hopeless; and it cannot represent two seasonalities
at once. The standard remedy is to move seasonality into exogenous Fourier
regressors and keep a low-order non-seasonal ARMA for the residual dependence.

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
reports mean ± std over repeats. Training and inference are timed separately;
`results/tables/hardware.json` records the exact machine.

## Configuration

Everything tunable lives in `configs/config.yaml` — nothing is hard-coded in the
scripts. To change the input window length, edit `forecasting.sequence_length`;
to change the evaluation week, edit `splits`.

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
- Results are single-seed unless `timing.n_repeats` is raised; for publication
  quality, average across seeds.

## Report and video

- Report: `report/` (submitted separately as PDF)
- Video: link in the report's References section

## Data citation

G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan
and the Province of Trentino," *Scientific Data*, vol. 2, 150055, 2015.
https://doi.org/10.1038/sdata.2015.55
