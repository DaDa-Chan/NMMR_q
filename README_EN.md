# NMMR-Q: Learning the Treatment Confounding Bridge Function via Neural Maximum Moment Restriction for Proximal Doubly Robust Causal Estimation

English | [中文](README.md)

## Overview

This project proposes **NMMR-Q**, which applies the Neural Maximum Moment Restriction (NMMR) framework to learn the treatment confounding bridge function $q$, and combines it with the existing NMMR-H to achieve proximal doubly robust (PDR) estimation.

**Core Contributions:**

1. **NMMR-Q method**: Derives the residual construction, kernel space, and dual-network softplus architecture adapted for $q$, enabling the proximal inverse probability weighted (PIPW) estimator
2. **Proximal doubly robust estimation**: Combines NMMR-Q with NMMR-H to implement the first complete PDR estimation pipeline within the neural KMMR framework, unifying PIPW, POR, and PDR under a single framework

## Project Structure

```
NMMR_q/
├── src/                          # Core source code
│   ├── experiment.py             #   Main experiment entry (single-model training + eval)
│   ├── models/
│   │   ├── NMMRq/                #   NMMR-Q: model, loss, trainers, experiments
│   │   ├── NMMRh/                #   NMMR-H: model, loss, trainers, experiments
│   │   └── LINEAR/               #   Linear baseline model
│   ├── data/ate/                 #   Data generation and loading
│   │   ├── sgd_pv.py             #     Synthetic data generation (true ATE = 2.0)
│   │   ├── data_class.py         #     SGDDataset / RHCDataset / BootstrapDataset
│   │   └── rhc_preprocess.py     #     RHC data preprocessing
│   └── utils.py                  #   Utility functions
├── scripts/                      # HPO and focused experiment scripts
│   ├── run_pilot_protocol.py     #   SGD pilot HPO entry point
│   └── analyze_rhc_repeated.py   #   Focused RHC repeated-CF analysis
├── configs/                      # JSON configuration files
│   ├── sgd/                      #   SGD dataset configs (nmmr_q / nmmr_h)
│   └── rhc/                      #   RHC dataset configs (nmmr_q / nmmr_h)
├── data/                         # Input data
│   ├── SGD/                      #   Synthetic data
│   └── right_heart_catheterization/  # RHC observational data
├── summary/                      # Experiment result summaries
│   ├── *.csv                     #   Summary CSV data files
│   └── figures/                  #   Generated plots (PNG + PDF)
├── predicts/                     # Model prediction outputs
├── train_results/                # Training logs
├── tune_results/                 # Hyperparameter tuning logs
├── report/                       # LaTeX report
│   ├── report.tex                #   English report
│   ├── report_cn.tex             #   Chinese report
│   └── references.bib            #   Bibliography
├── figures/                      # Early-stage analysis plots
├── double_rob.py                 # Doubly robust estimation entry point
├── main.py                       # Project main entry
└── requirements.txt              # Python dependencies
```

## Installation

**Requirements**: Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.2, scikit-learn 1.5, Optuna 4.6, TensorBoard 2.16.

## Data Preparation

### Synthetic Data (SGD)

Place `sgd_train.csv` and `sgd_cf.csv` in `data/SGD/`. To regenerate:

```bash
python src/data/ate/sgd_pv.py
```

Supports 4 scenarios controlling proxy variable nonlinearity. True ATE = 2.0.

### Right Heart Catheterization (RHC)

Place the following files in `data/right_heart_catheterization/`:

- `rhc_train.csv`, `rhc_val.csv`, `rhc_test.csv`
- `RHC_X_allfeatures_list.csv`, `RHC_X_significantfeatures_list.csv`

## Experiment Workflow

SGD uses a strict separation between pilot HPO and formal experiments. Run
`scripts/run_pilot_protocol.py`, inspect the candidate, and manually promote each passed
candidate to `configs/sgd_frozen/`. Formal Monte Carlo runs read those FrozenConfig files
without running Optuna again. The legacy SGD `*_fixed.json` files have been removed.

### Doubly Robust Estimation (PDR)

`double_rob.py` is the shared P0/E1/E2 entry point for H/Q fitting and PIPW, POR, and PDR.
The current SGD protocol supports `oracle`, `parametric_proximal`, `nmmr_u`, and `nmmr_v`
under S1. Only NMMR-U/V read the promoted FrozenConfig files. LS, LS-QF, and 2SLS have not
yet been connected, so the command below is a wiring smoke test rather than the complete P0
method comparison.

```bash
python double_rob.py \
  --dataset_name sgd --run_stage P0 --scenarios S1 \
  --methods oracle,parametric_proximal,nmmr_u,nmmr_v \
  --sample_sizes 300 --repetitions 1 --test_size 1000 --n_splits 5 \
  --base_seed 20261103 --output_dir results/p0_s1_smoke_v1
```

The SGD protocol does not support resume: every run must use a new output directory. Delete
an incomplete directory and restart the same command after an interruption. A failure of one
method is still recorded without stopping the remaining methods and repetitions.

The legacy RHC path remains available:

```bash
# RHC Bootstrap: resample with replacement for 95% CIs
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode bootstrap

# RHC Repeated: eliminate fold-split noise
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode repeated
```

### Formal Monte Carlo and reporting

The legacy log parsers and hard-coded analysis scripts have been removed. The formal
correctness experiment will write structured repetition- and fold-level CSV files directly;
new summary and plotting entry points will consume those files instead of parsing terminal
logs. The implementation sequence is recorded locally under `plan/p0_e1_e2/`, with
`plan/实验计划.md` as the master plan.

## Model Architecture

### NMMR-Q (Treatment Confounding Bridge) → PIPW Estimator

| Feature | Description |
|---------|-------------|
| Network | Dual networks: $q_0(Z,X)$ / $q_1(Z,X)$ learned separately |
| Activation | Hidden: LeakyReLU; Output: softplus (ensures $q > 0$) |
| Kernel space | Gaussian RBF kernel on $(W, X)$ |
| Loss | Kernel Maximum Moment Restriction (KMMR), U/V-statistic variants |

### NMMR-H (Outcome Confounding Bridge) → POR Estimator

| Feature | Description |
|---------|-------------|
| Network | Single network: $h(A, W, X)$ |
| Kernel space | Gaussian RBF kernel on $(A, Z, X)$ |
| ATE | $\hat{\tau} = n^{-1}\sum_i [h(1,W_i,X_i) - h(0,W_i,X_i)]$ |

### PDR (Proximal Doubly Robust)

$$\hat{\tau}_{\text{PDR}} = \frac{1}{n}\sum_i \left[(2A_i-1)\,q_{A_i}(Z_i,X_i)(Y_i - h(A_i,W_i,X_i)) + h(1,W_i,X_i) - h(0,W_i,X_i)\right]$$

Remains consistent when either $q$ or $h$ is correctly specified.

## Configuration Files

SGD uses two config roles:

- **Search configs** (e.g., `nmmr_q_v_sgd.json`): search ranges and DGP settings for pilot HPO.
- **FrozenConfig artifacts** under `configs/sgd_frozen/`: manually promoted, passed HPO
  candidates for standalone and the formal cross-fitting protocol.

Legacy SGD `*_fixed.json` files are no longer supported. RHC legacy fixed configs remain
temporarily for the not-yet-migrated RHC path.

| Config pattern | Dataset | Model | Loss |
|----------------|---------|-------|------|
| `nmmr_q_u_*.json` | SGD / RHC | NMMR-Q | U-statistic |
| `nmmr_q_v_*.json` | SGD / RHC | NMMR-Q | V-statistic |
| `nmmr_h_u_*.json` | SGD / RHC | NMMR-H | U-statistic |
| `nmmr_h_v_*.json` | SGD / RHC | NMMR-H | V-statistic |

## TensorBoard

```bash
tensorboard --logdir results/sgd/nmmr_q --port 6006
```

Set `"log_metrics": "True"` in the config to enable training logs.

## Key References

- Kompa, Bellot & Gasquez (2022). *Deep Learning Methods for Proximal Inference via Maximum Moment Restriction.* NeurIPS.
- Cui et al. (2024). *Semiparametric Proximal Causal Inference.* JASA.
- Tchetgen Tchetgen et al. (2020). *An Introduction to Proximal Causal Learning.* arXiv.
