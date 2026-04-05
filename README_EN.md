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
├── scripts/                      # Analysis and visualization scripts
│   ├── analysis.py               #   Statistical analysis and hypothesis testing
│   ├── generate_figures.py       #   Generate report figures (7 plots + summary table)
│   ├── parse_train.py            #   Parse training logs → CSV
│   └── parse_tune.py             #   Parse tuning logs → CSV
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

The project uses a **two-stage workflow**: find optimal hyperparameters via Optuna first, then run formal experiments with fixed parameters.

### Stage 1: Hyperparameter Optimization (HPO)

Use config files with search spaces to find optimal hyperparameters via K-fold cross-validation:

```bash
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd.json \
  --dump_folder results
```

### Stage 2: Formal Experiments with Fixed Parameters

Copy the best parameters from HPO into `_fixed.json` configs:

```bash
# Single-model experiments
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd_fixed.json \
  --dump_folder results
```

### Doubly Robust Estimation (PDR)

`double_rob.py` runs both NMMR-Q and NMMR-H jointly, computing PIPW, POR, and PDR estimators:

```bash
# SGD: 100 trials, each with freshly generated data
python double_rob.py --dataset_name sgd --trials 100 --n_samples 5000

# RHC Bootstrap: resample with replacement for 95% CIs
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode bootstrap

# RHC Repeated: eliminate fold-split noise
python double_rob.py --dataset_name rhc --trials 100 --rhc_mode repeated
```

### Analysis and Visualization

Analysis scripts are in the `scripts/` directory. All paths auto-resolve to the project root:

```bash
# Generate report figures (output to summary/figures/)
python scripts/generate_figures.py

# Statistical analysis and hypothesis testing (output to figures/)
python scripts/analysis.py

# Parse training/tuning logs to CSV
python scripts/parse_train.py
python scripts/parse_tune.py
```

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

Configs are located under `configs/<dataset>/<model>/` in two types:

- **Search configs** (e.g., `nmmr_q_v_sgd.json`): hyperparameters as dicts with search ranges, for HPO
- **Fixed configs** (e.g., `nmmr_q_v_sgd_fixed.json`): hyperparameters as scalar values, for formal experiments

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
