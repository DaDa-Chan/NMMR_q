# Summary results

Small result tables behind the numbers reported in the paper. The full raw results (about 12 GB, mostly per-repetition prediction files) are not part of the repository.

## `simulation/`

Simulations on scenario S1 (true ATE 2), code version `90f8232`, base seed `20261201`.

- `analysis/`: output of `scripts/analyze_v2_confirmatory.py` over all runs.
  - `e2_bias_table.csv`: bias, Monte Carlo standard error, SD, RMSE and $\sqrt n\times$bias of POR, PIPW and PDR, per method and sample size.
  - `e2_slopes.csv`: slope of log|bias| on log n with bootstrap intervals.
  - `e4_coverage.csv`: coverage of the 95% PDR interval, mean interval length, SE/SD.
  - `e3_bias_table.csv`, `e3_slopes.csv`: NMMR-V at n = 2000 against the batch size.
  - `e1_mechanism.csv`: SD ratio and maximum of the fitted treatment bridge, slope of the fitted outcome bridge in W.
- `per_repetition/<run>/`: one directory per run.
  - `effect_results.csv`: POR, PIPW and PDR estimates, PDR standard error and interval, and timings for every repetition and method.
  - `repetitions.csv`: seeds and data hashes of each repetition; repetitions with the same sample size and number share the same data across runs.
  - `run_config.json`: methods, sample sizes, number of repetitions, frozen-configuration hashes and code version.

  Runs: `v2_E2_n500_1000`, `v2_E2_n2000_CV`, `v2_E2_n2000_U`, `v2_E2_n5000_CV`, `v2_E2_n5000_U`, `v2_E2_n10000` (bias and coverage against n), and `v2_E3_b{16,32,64,128,256}` (NMMR-V at n = 2000 against the batch size).

## `rhc/`

Right heart catheterization application, code version `e602179`.

- `tuning/`: the four selected configurations (H-V, Q-V, H-U, Q-U) with their scores; the candidate pool is `configs/rhc/candidate_pool_v2.json`. Tuning seed `20261126`.
- `main/`: 20 repetitions of five-fold cross-fitting (estimation seed `20261127`).
  - `results.csv`: per repetition and method, POR, PIPW, PDR, PDR standard error and interval, clipping levels and fractions.
  - `combined.csv`: the 20 repetitions combined by the median rule.
  - `run_config.json`: configurations per fold and settings.
- `sensitivity/`: the same protocol on repetitions 0 to 4 with one setting changed: `c0.5`, `c1` (bandwidth multiplier of the NMMR-V outcome bridge, shared by NMMR-RW) and `b256`, `b1024` (batch size of all four configurations).

## Reproducing the paper's tables and figures

`paper/figures/make_paper_figures.py` builds the figures and tables of the simulation and application sections from the full results. The numeric tables of bias, RMSE, coverage, slopes and the RHC estimates can be recomputed from the files here. The bridge diagnostics (the E1 table, the residual checks on the RHC data) need the stored predictions, which are not included.
