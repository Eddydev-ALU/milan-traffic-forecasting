# Report structure and artefact map

This maps each section of my report to the files that generate its numbers and
figures, so every value in the write-up can be traced back to the script that
produced it. Nothing in the report was typed by hand from a console.

| Report section | Backing artefacts |
|---|---|
| 3.3 Memory evidence | `results/tables/memory_benchmark.csv`, `results/figures/fig00_memory.png` |
| 3.4 Full ingest | `results/tables/ingest_profile.csv`, `data/processed/meta.json` |
| 4.1 Spatial distribution | `results/tables/distribution_stats.json`, `results/figures/fig01_spatial_distribution.png` |
| 4.2 The five series | `results/tables/top_squares.csv`, `series_comparison.csv`, `square_locations.json`, `fig02_selected_series.png` |
| 4.3 ACF / PACF | `results/tables/timeseries_stats.json`, `results/figures/fig03_acf_pacf.png` |
| 4.3 Stationarity | `results/tables/stationarity.csv` |
| 4.3 MSTL decomposition | `results/tables/timeseries_stats.json`, `results/figures/fig04_mstl.png` |
| 4.3 Weekly profile, periodogram | `results/figures/fig05_profiles.png`, `fig06_periodogram.png` |
| 5.4 Tuning | `results/tables/experiment_log.csv` (56 trials, each with a rationale) |
| 6.1 Per-area metrics | `results/tables/metrics_sq<ID>.csv`, `metrics_all.csv` |
| 6.1 Prediction plots | `results/figures/fig_pred_sq<ID>_<MODEL>.png`, `fig_overlay_sq<ID>*.png` |
| 6.1 Cross-area comparison | `results/figures/fig_compare_{MAE,RMSE,WAPE_pct}.png` |
| 6.4 Timing | `results/tables/timing.csv`, `hardware.json` |
| 6.4 Training curves | `results/figures/fig_training_curves.png` |
| 6.5 Failure analysis | `results/tables/failure_analysis_sq<ID>.json`, `failure_summary_sq<ID>.csv`, `fig_error_breakdown_sq<ID>.png` |

## Section outline

1. **Introduction** — operational motivation, dataset in one paragraph, research
   question, contribution, headline finding.
2. **Related Work** — organised by approach: classical statistical forecasting,
   recurrent models for cellular traffic, convolutional and attention-based
   sequence models, spatio-temporal approaches. Closes with how the review
   shaped my design.
3. **Dataset and Data Preparation** — raw format, the country-code row
   explosion, the ingest optimisation, memory evidence, missing-data policy,
   and the trade-offs I accepted.
4. **Exploratory Analysis** — spatial distribution, the five required series
   with their real-world identification, then ACF/PACF, stationarity, MSTL and
   the periodogram on the busiest square.
5. **Methodology** — problem formulation, splits and preprocessing, the three
   models with justification, the baselines, and the tuning strategy.
6. **Results and Discussion** — per-area metrics, comparison against the
   baselines, how the winning model changes by area, computational trade-offs,
   and failure analysis.
7. **Conclusion and Future Work** — findings against the research question,
   limitations, extensions.
8. **References** — IEEE style, including the repository and video links.
