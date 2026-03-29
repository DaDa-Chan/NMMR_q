# NMMR-q: Neural Maximum Moment Restriction for Causal Inference

A causal inference framework based on Neural Maximum Moment Restriction (NMMR) for estimating Average Treatment Effects (ATE) from observational data. This project implements two estimators -- NMMR-q (Q-function) and NMMR-h (H-function) -- and supports Proximal Doubly Robust (PDR) estimation.

## Project Structure

```
NMMR_q/
├── src/
│   ├── experiment.py              # Main experiment entry point
│   ├── models/
│   │   ├── NMMRq/                 # Q-function: model, loss, trainers, experiments
│   │   ├── NMMRh/                 # H-function: model, loss, trainers, experiments
│   │   └── LINEAR/                # Linear baseline model
│   └── data/ate/                  # Data generation and loading
│       ├── sgd_pv.py              # Synthetic data generation (true ATE=2.0)
│       └── data_class.py          # SGDDataset / RHCDataset / BootstrapDataset
├── configs/                       # JSON configuration files
│   ├── sgd/                       # SGD dataset configs
│   └── rhc/                       # RHC dataset configs
├── double_rob.py                  # Doubly robust estimation experiments (PDR)
├── analysis.py                    # Statistical analysis script
├── data/                          # Input data
│   ├── SGD/                       # Synthetic data
│   └── right_heart_catheterization/  # RHC observational data
├── requirements.txt               # Python dependencies
└── README.md
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

Supports 4 scenarios (scenario 1-4) controlling proxy variable nonlinearity. True ATE = 2.0.

### Right Heart Catheterization (RHC)

Place the following files in `data/right_heart_catheterization/`:
- `rhc_train.csv`, `rhc_val.csv`, `rhc_test.csv`
- `RHC_X_allfeatures_list.csv`, `RHC_X_significantfeatures_list.csv`

Ensure `data_configs.data_path` in config files matches the actual data location.

## Experiment Workflow

The project uses a **two-stage workflow**: first find optimal hyperparameters via Optuna, then run formal experiments with fixed parameters.

### Stage 1: Hyperparameter Optimization (HPO)

Use config files with search spaces (e.g., `nmmr_q_v_sgd.json`) to find optimal hyperparameters via K-fold cross-validation:

```bash
# SGD
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd.json \
  --dump_folder results

# RHC
python src/experiment.py \
  --dataset_name rhc \
  --model_name nmmr_q \
  --config_path configs/rhc/nmmr_q/nmmr_q_v_rhc.json \
  --dump_folder results
```

Search space config example (hyperparameters declared as dict with search ranges):

```json
{
  "train_params": {
    "n_epochs": 150,
    "batch_size": 500,
    "learning_rate": {"type": "float", "low": 9e-4, "high": 5e-3, "log": true},
    "network_width": {"type": "categorical", "choices": [45, 50, 55]},
    "l2_penalty": {"type": "float", "low": 1e-6, "high": 3e-6, "log": true},
    "hpo_n_trials": 15
  }
}
```

After HPO completes, Optuna outputs the best parameter combination.

### Stage 2: Write Best Parameters to `_fixed.json`

Copy the optimal hyperparameters from HPO into the corresponding `_fixed.json` config. Fixed configs use scalar values (not search spaces) and skip Optuna entirely:

```json
{
  "train_params": {
    "n_epochs": 150,
    "batch_size": 500,
    "learning_rate": 0.008923,
    "network_width": 55,
    "l2_penalty": 2.137e-06
  }
}
```

### Stage 3: Formal Experiments

#### Single-Model Repeated Experiments

Use fixed configs via `experiment.py` (set `n_trials` in config to control repetitions):

```bash
# SGD: each trial generates fresh data + cross-fitting
python src/experiment.py \
  --dataset_name sgd \
  --model_name nmmr_q \
  --config_path configs/sgd/nmmr_q/nmmr_q_v_sgd_fixed.json \
  --dump_folder results

# RHC: cross-fitting on full dataset
python src/experiment.py \
  --dataset_name rhc \
  --model_name nmmr_q \
  --config_path configs/rhc/nmmr_q/nmmr_q_v_rhc_fixed.json \
  --dump_folder results
```

#### Doubly Robust Estimation (PDR)

`double_rob.py` automatically reads `_fixed.json` configs, runs both NMMR-q and NMMR-h, and computes the PDR estimator:

```bash
# SGD: each trial generates new data, iterates over 4 scenarios x 2 statistics
python double_rob.py \
  --dataset_name sgd \
  --trials 100 \
  --n_samples 2000

# RHC (repeated cross-fitting): same data, different seeds control fold splits
python double_rob.py \
  --dataset_name rhc \
  --trials 20 \
  --rhc_mode repeated

# RHC (bootstrap): resample with replacement, construct statistical CIs
python double_rob.py \
  --dataset_name rhc \
  --trials 200 \
  --rhc_mode bootstrap
```

PDR formula: `PDR = (-1)^(1-A) * q * (Y - h_fact) + (h_1 - h_0)`

### SGD vs RHC Experiment Differences

| | SGD | RHC |
|---|---|---|
| Data source | Freshly generated each trial | Fixed dataset (5735 samples) |
| Variation across trials | Data sampling + fold splits | Fold splits only (repeated) or resampling (bootstrap) |
| Repeated mode | - | Eliminates split noise, yields stable point estimate |
| Bootstrap mode | - | Approximates sampling distribution, constructs 95% CI |
| Scenarios | 1-4 (proxy nonlinearity) | None |

## Configuration Files

Config files are located under `configs/<dataset>/<model>/` and come in two types:

**Search configs** (for HPO stage):

| Config | Dataset | Model | Loss Type |
|--------|---------|-------|-----------|
| `nmmr_q_u_sgd.json` | SGD | NMMR-q | U-statistic |
| `nmmr_q_v_sgd.json` | SGD | NMMR-q | V-statistic |
| `nmmr_h_u_sgd.json` | SGD | NMMR-h | U-statistic |
| `nmmr_h_v_sgd.json` | SGD | NMMR-h | V-statistic |
| `nmmr_q_u_rhc.json` | RHC | NMMR-q | U-statistic |
| `nmmr_q_v_rhc.json` | RHC | NMMR-q | V-statistic |
| `nmmr_h_u_rhc.json` | RHC | NMMR-h | U-statistic |
| `nmmr_h_v_rhc.json` | RHC | NMMR-h | V-statistic |

**Fixed configs** (for formal experiments, `_fixed.json` suffix):

After HPO, copy the best parameters here. `double_rob.py` reads fixed configs by default. Every search config has a corresponding fixed version (e.g., `nmmr_q_v_sgd.json` -> `nmmr_q_v_sgd_fixed.json`).

## Model Architecture

### NMMR-q (Q-function) -> PIPW Estimator

- **Dual-network design**: Separate networks `q0(Z,X)` / `q1(Z,X)` for A=0 and A=1
- **Architecture**: Fully connected layers with LeakyReLU activation and softplus output
- **Loss function**: Kernel-based moment restriction loss using Gaussian RBF kernels on (W, X) space
- **Bandwidth selection**: Automatic estimation via median heuristic
- **Supports both U-statistic and V-statistic loss variants**

### NMMR-h (H-function) -> POR Estimator

- **Single-network design**: `h(A, W, X)` predicts outcome
- **Loss function**: Kernel defined on (A, Z, X) space
- **ATE prediction**: `E[h(1,W,X) - h(0,W,X)]`

### PDR (Proximal Doubly Robust)

Combines PIPW and POR. Even if one model is misspecified, as long as the other is correctly specified, PDR remains consistent.

## TensorBoard

Set `"log_metrics": "True"` in the config to enable training logs:

```bash
# View a single experiment
tensorboard --logdir results/sgd/nmmr_q/nmmr_q_v_sgd/<timestamp>/<seed>/tensorboard_log_<seed> --port 6006

# Compare multiple experiments
tensorboard --logdir results/sgd/nmmr_q --port 6006
```

Open `http://localhost:6006` in your browser to view loss curves and training metrics.

## Output Structure

```
results/<dataset>/<model>/<config>/<timestamp>/
├── <seed>/
│   ├── tensorboard_log_<seed>/    # TensorBoard event files
│   ├── fold_1/, fold_2/, ...      # Per-fold results
│   └── *.pred.txt                 # E[Y|do(A=0)], E[Y|do(A=1)], ATE
predicts/<dataset>/dr_results/     # Doubly robust experiment result CSVs
predicts/<dataset>/<model>/        # Single-model aggregated predictions
train_results/                     # Training result summaries
```
