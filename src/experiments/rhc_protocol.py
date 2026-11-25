"""
Stage 1 of the RHC analysis: cross-fitted POR/PIPW/PDR with the configurations
fixed in stage 0, repeated over split/initialization seeds, and combined with
the DML median rule.

Per repetition:

1. One A-stratified ``n_splits``-fold plan (``build_fold_plan``), shared by
   every method and both bridges.
2. For each outer fold, the RHC transform is fitted on that fold's training
   rows only and applied to all rows (``transformed_dataset``).
3. Each NMMR method fits H and Q fold by fold through the existing
   ``_run_cross_fitting`` (one-fold plans), so the training, checkpointing
   and early stopping are exactly those of the simulations. Out-of-fold
   predictions are assembled over the folds.
4. POR/PIPW/PDR from ``compute_effect_scores``; a 95% interval for PDR only
   (``compute_influence_ci``).

Methods: NMMR-V and NMMR-U use their own tuned configurations; NMMR-RW
(``nmmr_c``) uses the V configuration and NMMR-U+clip the U configuration,
with the same overrides as S1 (``nmmr_method_param_overrides``) except that
the H clipping level is fixed at ``H_CLIP_RHC`` (twice the outcome range).
The parametric proximal baseline is fitted on the same folds.

Results are appended per (repetition, method), so an interrupted run resumes
from the next missing pair when the run config matches.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.ate.rhc_protocol_data import (
    RHCFrameDataset, W_COLUMNS, X_COLUMNS, Z_COLUMNS, transformed_dataset,
)
from src.experiments.effects import compute_effect_scores, compute_influence_ci
from src.experiments.rhc_tuning import _git_commit, frame_fingerprint, load_selected
from src.experiments.sgd_experiment_protocol import (
    _derive_bridge_init_seed, build_fold_plan, nmmr_method_param_overrides,
)
from src.models.experiment_common import output_clip_levels
from src.utils import get_device, make_seed_bundle

NEURAL_METHODS = ("nmmr_v", "nmmr_u", "nmmr_c", "nmmr_u_clip")
ALL_METHODS = NEURAL_METHODS + ("parametric_proximal",)
CONFIG_SOURCE = {"nmmr_v": "V", "nmmr_c": "V", "nmmr_u": "U", "nmmr_u_clip": "U"}
H_CLIP_RHC = 60.0
DEFAULT_N_SPLITS = 5

RESULT_COLUMNS = [
    "repetition", "method", "n", "split_seed", "init_seed", "fold_plan_hash",
    "tau_por", "tau_pipw", "tau_pdr", "pdr_se", "pdr_ci_lower", "pdr_ci_upper", "pdr_score_var",
    "q_clip_level", "h_clip_level", "q_clip_frac", "h_clip_frac", "seconds", "failure_reason",
]


# ---------------------------------------------------------------------------
# Configurations and overrides
# ---------------------------------------------------------------------------

def load_stage0_configs(selected_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{"V": {"H": params, "Q": params}, "U": {...}} from stage 0's selected.json files."""
    selected_dir = Path(selected_dir)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for objective in ("V", "U"):
        out[objective] = {}
        for bridge in ("H", "Q"):
            task = f"{bridge}-{objective}"
            out[objective][bridge] = load_selected(selected_dir / task / "selected.json", task)["train_params"]
    return out


def rhc_param_overrides(method: str, n: int) -> Dict[str, Any]:
    """S1 overrides (diagonal weight, clipping) with the H clip level set for RHC."""
    overrides = nmmr_method_param_overrides(method, n)
    if "h_output_clip" in overrides:
        overrides["h_output_clip"] = H_CLIP_RHC
    return overrides


# ---------------------------------------------------------------------------
# DML median combination
# ---------------------------------------------------------------------------

def combine_repetitions(psi: Sequence[float], se: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    """
    Median rule of Chernozhukov et al. (2018):
        psi~ = median_s psi_s,   SE~^2 = median_s { SE_s^2 + (psi_s - psi~)^2 },
    interval psi~ +- z * SE~. Equal repetitions reduce to the single interval;
    dispersed point estimates widen it.
    """
    from scipy.stats import norm

    psi = np.asarray(psi, dtype=float)
    se = np.asarray(se, dtype=float)
    if psi.size == 0 or psi.size != se.size:
        raise ValueError("combine_repetitions: need equal, non-zero numbers of estimates and SEs.")
    if not (np.all(np.isfinite(psi)) and np.all(np.isfinite(se))):
        raise ValueError("combine_repetitions: non-finite input; drop failed repetitions first.")
    psi_med = float(np.median(psi))
    se_med = float(math.sqrt(np.median(se ** 2 + (psi - psi_med) ** 2)))
    z = float(norm.ppf(0.5 + level / 2))
    return {"psi": psi_med, "se": se_med, "ci_lower": psi_med - z * se_med,
            "ci_upper": psi_med + z * se_med, "n_repetitions": int(psi.size)}


# ---------------------------------------------------------------------------
# One method on one repetition
# ---------------------------------------------------------------------------

def _fold_datasets(frame: pd.DataFrame, splits, device) -> List[RHCFrameDataset]:
    return [transformed_dataset(frame, train_idx, device=device) for train_idx, _ in splits]


def fit_nmmr_method(method: str, fold_datasets: List[RHCFrameDataset], splits, split_seed: int,
                    rep_init_seed: int, fold_configs: List[Dict[str, Dict[str, Dict[str, Any]]]],
                    n: int) -> Dict[str, Any]:
    """
    Out-of-fold H and Q predictions of one NMMR method, fold by fold.
    ``fold_configs[k]`` is the configuration set used on outer fold k (the
    same set on every fold in the main analysis; fold-specific sets in the
    strict nested sensitivity analysis).
    """
    from src.models.NMMRh.nmmr_h_experiments import _run_cross_fitting as run_cf_h
    from src.models.NMMRq.nmmr_q_experiments import _run_cross_fitting as run_cf_q

    if len(fold_configs) != len(splits):
        raise ValueError(f"Need one configuration set per fold; got {len(fold_configs)} for {len(splits)} folds.")
    overrides = rhc_param_overrides(method, n)
    q_base = _derive_bridge_init_seed(rep_init_seed, method, "Q")
    h_base = _derive_bridge_init_seed(rep_init_seed, method, "H")

    q_obs = np.full(n, np.nan)
    h_obs = np.full(n, np.nan)
    h0 = np.full(n, np.nan)
    h1 = np.full(n, np.nan)
    diagnostics: List[Dict[str, Any]] = []
    q_params = h_params = {}
    for k, ((train_idx, eval_idx), ds) in enumerate(zip(splits, fold_datasets), start=1):
        source = fold_configs[k - 1][CONFIG_SOURCE[method]]
        q_params = {**source["Q"], **overrides}
        h_params = {**source["H"], **overrides}
        # One-fold plans: pass a fold-specific loader seed and init seed so that
        # folds do not share them (the helper numbers every one-fold call as fold 1).
        common = dict(dataset=ds, n_splits=len(splits), data_configs={}, random_seed=split_seed + k - 1,
                      log_folder=None, fold_plan=[(train_idx, eval_idx)])
        _, _, q_full, q_diag, _ = run_cf_q(data_name="rhc", train_params=q_params, tag=f"{method}_q_f{k}",
                                           init_seed=make_seed_bundle(q_base, k).init_seed, **common)
        _, _, h_full, h_diag, _ = run_cf_h(data_name="rhc", train_params=h_params, tag=f"{method}_h_f{k}",
                                           init_seed=make_seed_bundle(h_base, k).init_seed, **common)
        q_obs[eval_idx] = np.asarray(q_full).reshape(-1)[eval_idx]
        h_obs[eval_idx] = np.asarray(h_full["h_fact"]).reshape(-1)[eval_idx]
        h0[eval_idx] = np.asarray(h_full["h_0"]).reshape(-1)[eval_idx]
        h1[eval_idx] = np.asarray(h_full["h_1"]).reshape(-1)[eval_idx]
        for bridge, diag in (("Q", q_diag), ("H", h_diag)):
            for d in diag:
                diagnostics.append({**d, "fold": k, "bridge": bridge})
    return {"q_observed": q_obs, "h_observed": h_obs, "h0": h0, "h1": h1,
            "q_clip_level": q_params.get("q_output_clip"), "h_clip_level": h_params.get("h_output_clip"),
            "fold_diagnostics": diagnostics}


def fit_parametric(fold_datasets: List[RHCFrameDataset], splits, n: int, q_form: str = "arm_specific") -> Dict[str, Any]:
    """Parametric proximal bridges (Cui et al.) fitted on the same folds and transforms."""
    from src.models.LINEAR.linear_model import ProximalLinearModel

    q_obs = np.full(n, np.nan)
    h_obs = np.full(n, np.nan)
    h0 = np.full(n, np.nan)
    h1 = np.full(n, np.nan)
    diagnostics = []
    for k, ((train_idx, eval_idx), ds) in enumerate(zip(splits, fold_datasets), start=1):
        df = ds.data
        model = ProximalLinearModel(df.iloc[train_idx], z_cols=Z_COLUMNS, w_cols=W_COLUMNS,
                                    x_cols=X_COLUMNS, q_form=q_form)
        b_hat, used_lstsq = model.solve_h()
        t_hat, converged, message, residual = model.solve_q()
        if not converged:
            raise RuntimeError(f"Parametric q did not converge in fold {k}: {message}")
        W, Z, X = model.arrays(df.iloc[eval_idx])
        A = df["A"].to_numpy(dtype=float)[eval_idx]
        h_obs[eval_idx] = model.h_func(b_hat, W, A, X)
        h0[eval_idx] = model.h_func(b_hat, W, np.zeros_like(A), X)
        h1[eval_idx] = model.h_func(b_hat, W, np.ones_like(A), X)
        q_obs[eval_idx] = model.q_func(t_hat, Z, A, X)
        diagnostics.append({"fold": k, "h_used_lstsq": bool(used_lstsq), "q_residual_norm": residual})
    return {"q_observed": q_obs, "h_observed": h_obs, "h0": h0, "h1": h1,
            "q_clip_level": None, "h_clip_level": None, "fold_diagnostics": diagnostics}


def effect_row(frame: pd.DataFrame, preds: Dict[str, Any], por_only: bool = False) -> Dict[str, Any]:
    """
    POR/PIPW/PDR and the PDR interval from out-of-fold predictions. With
    ``por_only`` (the parametric baseline, whose cross-fitted arm-specific q
    is not usable; PIPW/PDR of the parametric baseline are on hold), PIPW,
    PDR and the interval are left as NaN. Any non-finite reported estimate
    raises, so it is recorded as a failure rather than written as inf.
    """
    if por_only:
        h1, h0 = np.asarray(preds["h1"], float), np.asarray(preds["h0"], float)
        if not (np.all(np.isfinite(h1)) and np.all(np.isfinite(h0))):
            raise ValueError("Non-finite or missing out-of-fold predictions in h0/h1.")
        nan = float("nan")
        return {"tau_por": float(np.mean(h1 - h0)), "tau_pipw": nan, "tau_pdr": nan, "pdr_se": nan,
                "pdr_ci_lower": nan, "pdr_ci_upper": nan, "pdr_score_var": nan, "q_clip_level": None,
                "h_clip_level": None, "q_clip_frac": nan, "h_clip_frac": nan}
    for key in ("q_observed", "h_observed", "h0", "h1"):
        if not np.all(np.isfinite(preds[key])):
            raise ValueError(f"Non-finite or missing out-of-fold predictions in {key}.")
    eff = compute_effect_scores(frame["Y"].to_numpy(float), frame["A"].to_numpy(float),
                                preds["q_observed"], preds["h_observed"], preds["h1"], preds["h0"])
    ci = compute_influence_ci(eff["pdr_scores"])
    row = {"tau_por": eff["tau_por"], "tau_pipw": eff["tau_pipw"], "tau_pdr": eff["tau_pdr"],
           "pdr_se": ci["se"], "pdr_ci_lower": ci["ci_lower"], "pdr_ci_upper": ci["ci_upper"],
           "pdr_score_var": float(np.var(eff["pdr_scores"])),
           "q_clip_level": preds["q_clip_level"], "h_clip_level": preds["h_clip_level"],
           "q_clip_frac": float("nan"), "h_clip_frac": float("nan")}
    reported = [row[k] for k in ("tau_por", "tau_pipw", "tau_pdr", "pdr_se")]
    if not np.all(np.isfinite(reported)):
        raise ValueError(f"Non-finite effect estimate (POR, PIPW, PDR, SE) = {reported}.")
    if preds["q_clip_level"] is not None:
        row["q_clip_frac"] = float(np.mean(preds["q_observed"] >= preds["q_clip_level"] - 1e-6))
    if preds["h_clip_level"] is not None:
        h_all = np.concatenate([preds["h0"], preds["h1"]])
        row["h_clip_frac"] = float(np.mean(np.abs(h_all) >= preds["h_clip_level"] - 1e-6))
    return row


# ---------------------------------------------------------------------------
# The whole stage
# ---------------------------------------------------------------------------

def repetition_seeds(base_seed: int, repetition: int):
    return make_seed_bundle(base_seed, repetition)


def run_rhc_protocol(frame: pd.DataFrame, out_dir: Path, base_seed: int, repetitions: Sequence[int],
                     methods: Sequence[str], selected_dir: Optional[Path] = None,
                     configs: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
                     fold_selected_dirs: Optional[Sequence[Path]] = None,
                     n_splits: int = DEFAULT_N_SPLITS, parametric_q_form: str = "arm_specific",
                     device: Any = None, log: Callable[[str], None] = print) -> pd.DataFrame:
    """
    Run the given repetitions and methods; results go to ``results.csv``,
    out-of-fold predictions to ``predictions/rep{r:03d}/{method}.npz`` and fold
    diagnostics to ``predictions/rep{r:03d}/{method}_folds.csv``. Returns the results table.
    """
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; expected a subset of {ALL_METHODS}.")
    fold_configs = None
    if any(m in NEURAL_METHODS for m in methods):
        if fold_selected_dirs is not None:
            if len(fold_selected_dirs) != n_splits:
                raise ValueError("fold_selected_dirs needs one directory per outer fold.")
            fold_configs = [load_stage0_configs(d) for d in fold_selected_dirs]
        else:
            if configs is None:
                if selected_dir is None:
                    raise ValueError("NMMR methods need stage 0's selected configurations (selected_dir).")
                configs = load_stage0_configs(selected_dir)
            fold_configs = [configs] * n_splits
    device = device if device is not None else get_device()
    frame = frame.reset_index(drop=True)
    n = len(frame)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "base_seed": base_seed, "n_splits": n_splits, "n": n, "frame_sha256": frame_fingerprint(frame),
        "fold_configs": fold_configs, "config_source": CONFIG_SOURCE, "h_clip_rhc": H_CLIP_RHC,
        "q_clip_level": output_clip_levels(n)[0], "parametric_q_form": parametric_q_form,
    }
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous.pop("git_commit", None)
        if previous != json.loads(json.dumps(run_config)):
            raise ValueError(f"{config_path} exists with a different run config; use a new output directory.")
    else:
        config_path.write_text(json.dumps({**run_config, "git_commit": _git_commit()}, indent=2))

    results_path = out_dir / "results.csv"
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = {(int(r.repetition), r.method) for r in prev.itertuples()}

    raw = RHCFrameDataset(frame)
    for rep in repetitions:
        todo = [m for m in methods if (int(rep), m) not in done]
        if not todo:
            continue
        seeds = repetition_seeds(base_seed, int(rep))
        plan = build_fold_plan(raw, n_splits, seeds.split_seed)
        fold_ds = _fold_datasets(frame, plan.splits, device)
        for method in todo:
            t0 = time.monotonic()
            row = {"repetition": int(rep), "method": method, "n": n, "split_seed": seeds.split_seed,
                   "init_seed": seeds.init_seed, "fold_plan_hash": plan.fold_plan_hash, "failure_reason": ""}
            try:
                if method == "parametric_proximal":
                    preds = fit_parametric(fold_ds, plan.splits, n, parametric_q_form)
                else:
                    preds = fit_nmmr_method(method, fold_ds, plan.splits, seeds.split_seed, seeds.init_seed,
                                            fold_configs, n)
                row.update(effect_row(frame, preds, por_only=(method == "parametric_proximal")))
                pred_dir = out_dir / "predictions" / f"rep{int(rep):03d}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.savez(pred_dir / f"{method}.npz", **{k: preds[k] for k in ("q_observed", "h_observed", "h0", "h1")})
                pd.DataFrame(preds["fold_diagnostics"]).assign(repetition=int(rep), method=method).to_csv(
                    pred_dir / f"{method}_folds.csv", index=False)
            except Exception as exc:  # recorded, never silently dropped
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.monotonic() - t0
            pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(results_path, mode="a",
                                                               header=not results_path.exists(), index=False)
            log(f"[rep {rep} {method}] PDR={row.get('tau_pdr', float('nan')):.3f} "
                f"({row['seconds']:.0f}s){' FAIL ' + row['failure_reason'] if row['failure_reason'] else ''}")
    return pd.read_csv(results_path)


def combine_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    DML median combination of PDR per method over successful repetitions
    (POR median only for the parametric baseline).
    """
    rows = []
    ok = results[results["failure_reason"].isna() | (results["failure_reason"].astype(str) == "")]
    for method, group in ok.groupby("method"):
        if group["tau_pdr"].notna().all():
            combined = combine_repetitions(group["tau_pdr"], group["pdr_se"])
        else:
            combined = {"psi": float("nan"), "se": float("nan"), "ci_lower": float("nan"),
                        "ci_upper": float("nan"), "n_repetitions": int(len(group))}
        rows.append({"method": method, **combined,
                     "n_failed": int(((results.method == method) & ~results.index.isin(ok.index)).sum()),
                     "por_median": float(group["tau_por"].median()),
                     "pipw_median": float(group["tau_pipw"].median()),
                     "pdr_midpoint_sd": float(group["tau_pdr"].std(ddof=1)) if len(group) > 1 else float("nan"),
                     "mean_halfwidth": float(((group["pdr_ci_upper"] - group["pdr_ci_lower"]) / 2).mean())})
    return pd.DataFrame(rows)
