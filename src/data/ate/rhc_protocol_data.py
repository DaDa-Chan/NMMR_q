"""
Right Heart Catheterization (RHC) data for the proximal analysis.

The variable roles and the covariate coding follow the proximal RHC
analyses of Tchetgen Tchetgen et al. (2020) and Cui et al. (2024):

* A = swang1 (1 = RHC), Y = t3d30 (days survived within 30 days, raw scale);
* Z = (pafi1, paco21), W = (ph1, hema1);
* X = the remaining 67 baseline covariates, with categorical variables
  dummy-coded against a fixed reference level (see ``X_COLUMNS``).

The column list and reference levels are taken from the public analysis
script of Li et al. (github.com/KenLi93/p2sls_surv_manuscript,
``rhc_data_analysis.R``), which uses the same Z, W and the same 67 X.

This module only builds the analysis frame from ``rhc_raw.csv``. Nothing
here depends on a train/evaluation split, so it can run before splitting.
Sample-dependent transforms (standardization, optional log transforms)
are in ``fit_rhc_transform`` / ``apply_rhc_transform`` and must be fitted
on a training part only.

The older ``rhc_preprocess.py`` (inherited from Kompa et al.) is not used
here: it log-transforms crea1/bili1 on the train split only and then
applies the train statistics to untransformed val/test columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

RHC_RAW_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "right_heart_catheterization" / "rhc_raw.csv"
)

A_COLUMN = "A"
Y_COLUMN = "Y"
Z_COLUMNS: List[str] = ["pafi1", "paco21"]
W_COLUMNS: List[str] = ["ph1", "hema1"]

X_COLUMNS: List[str] = [
    "age", "sex", "raceblack", "raceother", "edu", "income1", "income2", "income3",
    "ins_care", "ins_pcare", "ins_caid", "ins_no", "ins_carecaid", "cat1_copd",
    "cat1_mosfsep", "cat1_mosfmal", "cat1_chf", "cat1_coma", "cat1_cirr",
    "cat1_lung", "cat1_colon", "cat2_mosfsep", "cat2_coma", "cat2_mosfmal",
    "cat2_lung", "cat2_cirr", "cat2_colon", "resp", "card", "neuro", "gastr",
    "renal", "meta", "seps", "trauma", "ortho", "das2d3pc", "dnr1", "ca_yes",
    "ca_meta", "surv2md1", "aps1", "scoma1", "wtkilo1", "temp1", "meanbp1", "resp1",
    "hrt1", "wblc1", "sod1", "pot1", "crea1", "bili1", "alb1", "cardiohx", "chfhx",
    "dementhx", "psychhx", "chrpulhx", "renalhx", "liverhx", "gibledhx", "malighx",
    "immunhx", "transhx", "amihx", "wt0",
]

# Continuous covariates in X (everything else in X is a 0/1 indicator).
CONTINUOUS_X_COLUMNS: List[str] = [
    "age", "edu", "das2d3pc", "surv2md1", "aps1", "scoma1", "wtkilo1", "temp1",
    "meanbp1", "resp1", "hrt1", "wblc1", "sod1", "pot1", "crea1", "bili1", "alb1",
]

# Right-skewed lab values that may be log-transformed before standardizing.
LOG_CANDIDATE_COLUMNS: List[str] = ["crea1", "bili1"]


def _indicator(series: pd.Series, level: str) -> pd.Series:
    """1.0 where the value equals ``level``; missing values count as 0."""
    return (series == level).astype(float)


def load_rhc_raw(path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_csv(Path(path) if path is not None else RHC_RAW_PATH)


def build_rhc_frame(raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build the analysis frame: columns A, Y, Z_COLUMNS, W_COLUMNS, X_COLUMNS,
    all float, one row per patient, in the raw file's row order.

    Reference levels: income "Under $11k", insurance "Private", cat1 "ARF",
    cat2 missing, ca "No", race "white".
    """
    raw = load_rhc_raw() if raw is None else raw
    out = pd.DataFrame(index=raw.index)

    out[A_COLUMN] = _indicator(raw["swang1"], "RHC")
    out[Y_COLUMN] = raw["t3d30"].astype(float)
    for col in Z_COLUMNS + W_COLUMNS:
        out[col] = raw[col].astype(float)

    x = pd.DataFrame(index=raw.index)
    x["sex"] = _indicator(raw["sex"], "Female")
    x["raceblack"] = _indicator(raw["race"], "black")
    x["raceother"] = _indicator(raw["race"], "other")
    x["income1"] = _indicator(raw["income"], "$11-$25k")
    x["income2"] = _indicator(raw["income"], "$25-$50k")
    x["income3"] = _indicator(raw["income"], "> $50k")
    x["ins_care"] = _indicator(raw["ninsclas"], "Medicare")
    x["ins_pcare"] = _indicator(raw["ninsclas"], "Private & Medicare")
    x["ins_caid"] = _indicator(raw["ninsclas"], "Medicaid")
    x["ins_no"] = _indicator(raw["ninsclas"], "No insurance")
    x["ins_carecaid"] = _indicator(raw["ninsclas"], "Medicare & Medicaid")
    cat_levels = {
        "copd": "COPD", "mosfsep": "MOSF w/Sepsis", "mosfmal": "MOSF w/Malignancy",
        "chf": "CHF", "coma": "Coma", "cirr": "Cirrhosis", "lung": "Lung Cancer",
        "colon": "Colon Cancer",
    }
    for suffix, level in cat_levels.items():
        x[f"cat1_{suffix}"] = _indicator(raw["cat1"], level)
    for suffix in ["mosfsep", "coma", "mosfmal", "lung", "cirr", "colon"]:
        x[f"cat2_{suffix}"] = _indicator(raw["cat2"], cat_levels[suffix])
    for col in ["resp", "card", "neuro", "gastr", "renal", "meta", "seps", "trauma", "ortho", "dnr1"]:
        x[col] = _indicator(raw[col], "Yes")
    x["ca_yes"] = _indicator(raw["ca"], "Yes")
    x["ca_meta"] = _indicator(raw["ca"], "Metastatic")
    x["wt0"] = (raw["wtkilo1"] == 0).astype(float)
    for col in CONTINUOUS_X_COLUMNS:
        x[col] = raw[col].astype(float)
    for col in ["cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx",
                "liverhx", "gibledhx", "malighx", "immunhx", "transhx", "amihx"]:
        x[col] = raw[col].astype(float)

    missing = [c for c in X_COLUMNS if c not in x.columns]
    if missing:
        raise RuntimeError(f"X columns not constructed: {missing}")
    out = pd.concat([out, x[X_COLUMNS]], axis=1)
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"RHC analysis frame has missing values in {bad}")
    return out


# ---------------------------------------------------------------------------
# Sample-dependent transforms: fit on a training part, apply anywhere.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RhcTransform:
    """Location/scale per column, plus which columns were log-transformed first."""
    log_columns: List[str]
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)


def default_scaled_columns() -> List[str]:
    """Z, W and the continuous covariates. Y, A and indicators are left as is."""
    return Z_COLUMNS + W_COLUMNS + CONTINUOUS_X_COLUMNS


def fit_rhc_transform(
    train_df: pd.DataFrame,
    log_columns: Sequence[str] = tuple(LOG_CANDIDATE_COLUMNS),
    scaled_columns: Optional[Sequence[str]] = None,
) -> RhcTransform:
    """
    Estimate the transform from ``train_df`` only. Columns in
    ``log_columns`` are log-transformed before their mean and standard
    deviation are computed (they must be strictly positive).
    """
    scaled = list(default_scaled_columns() if scaled_columns is None else scaled_columns)
    logs = [c for c in log_columns if c in scaled]
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for col in scaled:
        values = train_df[col].to_numpy(dtype=float)
        if col in logs:
            if np.any(values <= 0):
                raise ValueError(f"Cannot log-transform {col}: non-positive values in the training part.")
            values = np.log(values)
        sd = float(np.std(values, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Column {col} has zero or non-finite spread in the training part.")
        means[col] = float(np.mean(values))
        stds[col] = sd
    return RhcTransform(log_columns=logs, means=means, stds=stds)


def apply_rhc_transform(df: pd.DataFrame, transform: RhcTransform) -> pd.DataFrame:
    """Return a copy of ``df`` with the fitted transform applied."""
    out = df.copy()
    for col, mean in transform.means.items():
        values = out[col].to_numpy(dtype=float)
        if col in transform.log_columns:
            if np.any(values <= 0):
                raise ValueError(f"Cannot log-transform {col}: non-positive values.")
            values = np.log(values)
        out[col] = (values - mean) / transform.stds[col]
    return out


# ---------------------------------------------------------------------------
# Tensor dataset for the NMMR trainers
# ---------------------------------------------------------------------------

class RHCFrameDataset(Dataset):
    """
    Torch dataset over an (already transformed) RHC analysis frame, with the
    attribute layout the NMMR trainers and cross-fitting helpers read:
    ``A``, ``Y`` of shape (n, 1), ``Z``, ``W``, ``X`` of shape (n, d), and
    ``data`` (the frame itself, for the parametric baseline). There is no
    latent ``U`` in real data; ``U`` is a zero column so batches have the
    same keys as the simulation datasets.
    """

    def __init__(self, df: pd.DataFrame, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32,
                 z_cols: Sequence[str] = tuple(Z_COLUMNS), w_cols: Sequence[str] = tuple(W_COLUMNS),
                 x_cols: Sequence[str] = tuple(X_COLUMNS)):
        self.data = df.reset_index(drop=True)
        self.z_cols, self.w_cols, self.x_cols = list(z_cols), list(w_cols), list(x_cols)

        def tensor(values: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(values, dtype=float), dtype=dtype, device=device)

        self.A = tensor(self.data[A_COLUMN].to_numpy()).reshape(-1, 1)
        self.Y = tensor(self.data[Y_COLUMN].to_numpy()).reshape(-1, 1)
        self.Z = tensor(self.data[self.z_cols].to_numpy())
        self.W = tensor(self.data[self.w_cols].to_numpy())
        self.X = tensor(self.data[self.x_cols].to_numpy())
        self.U = torch.zeros_like(self.A)

    def __len__(self) -> int:
        return self.A.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx], "A": self.A[idx], "Z": self.Z[idx], "W": self.W[idx],
                "U": self.U[idx], "Y": self.Y[idx]}


def transformed_dataset(frame: pd.DataFrame, fit_rows: Sequence[int],
                        device: str | torch.device = "cpu") -> RHCFrameDataset:
    """
    Fit the transform on ``frame.iloc[fit_rows]`` only and return a dataset
    over the WHOLE frame (same row order) with that transform applied. Rows
    outside ``fit_rows`` never influence the transform.
    """
    transform = fit_rhc_transform(frame.iloc[list(fit_rows)])
    return RHCFrameDataset(apply_rhc_transform(frame, transform), device=device)
