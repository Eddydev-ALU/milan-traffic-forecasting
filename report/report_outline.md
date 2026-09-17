# Report skeleton

Write the report from this outline. Every `[...]` is a placeholder; every
artefact reference points at a file one of the scripts generates, so nothing
needs retyping by hand. Target length: 8–12 pages including figures.

Marks are concentrated in interpretation, not in code: EDA (13) + time-series
analysis (12) + model design (14) + evaluation (13) + failure analysis (8) = 60
of 100. Budget your writing time accordingly.

---

## 1. Introduction  (~0.5 page)

- Why short-horizon cellular traffic forecasting matters operationally:
  radio-resource allocation, base-station energy saving, dynamic capacity
  planning, anomaly detection.
- The dataset in one paragraph: Milan, 10,000 squares, 10-minute slots, two months.
- State the research question verbatim from the brief, then your contribution:
  three architecturally distinct models compared on identical splits, scaling
  and metrics, across three areas with different traffic characteristics.
- One sentence previewing the headline finding.

## 2. Related Work  (~1.5 pages)

Organise by *approach*, not by paper — a numbered list of summaries reads as a
literature dump and scores poorly.

- Classical statistical forecasting (ARIMA/SARIMA and multi-seasonal variants);
  where it holds up and where it breaks on 10-minute data.
- Recurrent models for cellular traffic (LSTM/GRU), including work on this
  exact dataset.
- Convolutional and attention-based sequence models (TCN; the critical line of
  work questioning whether transformers beat simple baselines on long-horizon
  forecasting).
- Spatio-temporal approaches, and why this study is deliberately per-area.

Close with an explicit paragraph: *what this review changed about my design.*
That sentence is what connects Section 2 to Section 5 in the marker's mind.

## 3. Dataset and Data Preparation  (~1.5 pages) → Data Handling, 10 pts

- Raw format table; explain the country-code row explosion.
- Optimisation strategy: streaming one day at a time, `usecols`, narrow dtypes,
  aggregate-on-arrival via `np.add.at`, pre-allocated float32 matrix, `.npy`
  persistence with memory mapping downstream.
- **Table 1** ← `results/tables/memory_benchmark.csv`: naive vs optimised
  DataFrame footprint, peak Python allocation, wall time, reduction %, and the
  checksum confirming identical aggregates. Say that the benchmark is one day
  extrapolated to 62 — don't let the reader assume otherwise.
- **Figure 0** ← `results/figures/fig00_memory.png`.
- Full-ingest numbers ← `results/tables/ingest_profile.csv`: peak RSS, total
  wall time, final on-disk size.
- Missing-data policy and the fraction affected ← `meta.json`. Justify the
  interpolate-vs-zero choice.
- **Trade-offs paragraph** (do not skip; the rubric asks for it): country
  structure discarded, float32 precision, CPU time traded for RAM.

## 4. Exploratory Analysis  (~2.5 pages) → EDA 13 pts + Time-Series 12 pts

### 4.1 Spatial distribution
- **Figure 1** ← `fig01_spatial_distribution.png` (histogram, ECDF, grid map).
- Quantify the skew from `distribution_stats.json`: skewness, top-1%/5%/10%
  traffic share, Gini, log10 range. Prose like "heavily skewed" without numbers
  scores as description, not analysis.
- Connect forward: this is *why* per-area scaling is needed and *why* MAPE is
  unstable in low-traffic squares.

### 4.2 The five required series
- **Figure 2** ← `fig02_selected_series.png` (top-3 plus 4159 and 4556).
- **Table 2** ← `series_comparison.csv`: burstiness, weekend/weekday ratio,
  peak hour, night/peak ratio.
- Discuss *shape* differences, not just level: commuter-driven city-centre
  profile vs residential evening peak vs flat-with-spikes venue profile.
- Note that 1 Nov 2013 is All Saints' Day and 2–3 Nov a weekend, so the series
  opens with an atypical stretch.
- **Identify what 4159 and 4556 actually are.** Compute centroids from
  `milano-grid.geojson` and look them up. Naming the real location and tying it
  to the observed spikes is the cheapest available upgrade from a generic
  caption to a "meaningful real-world explanation".

### 4.3 Two further analyses on the busiest area
- **Figure 3** ← `fig03_acf_pacf.png`. Read off lag-1 correlation, the 144 and
  1008 peaks, and where the PACF cuts off. **State the window length L this
  implies** — this sentence is the bridge between Section 4 and Section 5, and
  it is the one most submissions omit.
- **Figure 4** ← `fig04_mstl.png` plus the variance shares. How much of the
  signal is pure calendar structure (i.e. free), and how much is left for a
  model to earn?
- **Table 3** ← `stationarity.csv`. ADF and KPSS test opposite nulls; say what
  their agreement or disagreement implies for differencing.
- Optionally **Figure 5/6** ← weekly profile heatmap, periodogram.
- Anomalies from the MSTL remainder: list the flagged timestamps and offer
  explanations where you can.

## 5. Methodology  (~2 pages) → Model Design 14 pts + Experimentation 10 pts

### 5.1 Problem formulation
Define x_a(t) and the one-step-ahead task formally. State the splits with dates
and step counts (train 1 Nov–8 Dec, val 9–15 Dec, test 16–22 Dec = 1,008 steps),
and say explicitly that splits are chronological and never shuffled.

### 5.2 Input representation (Task 4-VI — a required item)
Sequence length L and the ACF evidence for it; `log1p` then standardisation
fitted on training data only; inverse-transform before every metric; the six
calendar features and why sin/cos rather than raw hour.

### 5.3 The three models
For each: structure, why it suits *these* data characteristics, supporting
citation, and an honest limitation.
- **SARIMAX + Fourier** — explain the s=144 infeasibility and the Fourier
  remedy. This is a good candidate for the "one important technical decision"
  in your video.
- **LSTM** — gated recurrence for nonlinear regime switching.
- **TCN** — dilated causal convolutions, receptive field 1 + 2(k−1)(2^n − 1),
  parallel over time.
- **Baselines** — persistence and seasonal naive. State up front that
  persistence is strong at this horizon.

### 5.4 Tuning strategy
- **Table 4** ← `results/tables/experiment_log.csv`. **Fill in the `rationale`
  column by hand.** The brief demands documented reasoning for each successive
  adjustment; an unfilled column turns systematic experimentation back into
  "I ran a grid search".
- Early stopping on validation MAE in original units; test week never touched
  during selection.

## 6. Results and Discussion  (~3 pages) → Evaluation 13 pts + Failure 8 pts

- **Tables 5–7** ← `metrics_sq<ID>.csv`, one per area (Task 4-III).
- **Figures** ← the nine `fig_pred_sq<ID>_<MODEL>.png` (Task 4-II), plus the
  overlays and `fig_compare_*.png`.
- **Table 8** ← `timing.csv` and `hardware.json` (Task 4-IV). State the repeat
  count, the warm-up policy, whether times come from one area or an average,
  and the exact hardware.
- Cross-area comparison: does the ranking hold? Relate any change back to the
  Section 4 characteristics of each area.
- Training-time vs accuracy trade-off — a model 40× slower for 3% better MAE is
  a real finding about operational suitability.
- **Failure analysis** ← `failure_analysis_sq<ID>.json` and
  `fig_error_breakdown_sq<ID>.png`:
  - error by hour and by weekday;
  - the lag diagnostic — if `best_backshift > 0` the model is reproducing the
    previous observation rather than anticipating the next, and you can now say
    so with a number instead of a hand-wave at a plot;
  - one zoomed worst-period figure with a mechanistic explanation.
- Compare against persistence explicitly. If a neural model loses, report it.

## 7. Conclusion and Future Work  (~0.5 page)

Findings tied to the research question; limitations (single seed, per-area
modelling, country structure discarded, one week of test data); extensions
(spatio-temporal models, multi-step horizons, cross-area transfer).

## 8. References

IEEE style. Include the GitHub repository URL and the video link here, as the
brief requires.

---

## Pre-submission checklist

| Requirement | Where |
|---|---|
| Memory evidence before/after | Table 1, Figure 0 |
| Distribution figure + discussion | Figure 1, §4.1 |
| Top-3 areas identified | `top_squares.csv` |
| 5 series, first two weeks | Figure 2 |
| Two extra analyses | §4.3 |
| 3 models justified | §5.3 |
| 9 prediction plots | `fig_pred_sq*_*.png` |
| 3 metric tables (MAE/MAPE/RMSE) | Tables 5–7 |
| Timing + hardware + method | Table 8 |
| Input representation stated | §5.2 |
| Comparative analysis | §6 |
| Failure case | §6 |
| Iterative tuning documented | Table 4, rationale column filled |
| GitHub link | §8 |
| Video link | §8 |
