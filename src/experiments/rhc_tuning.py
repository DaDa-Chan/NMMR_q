"""
Stage 0 of the RHC analysis: choose one configuration per tuning task from a
fixed candidate pool, by held-out U moment risk.

Design (all choices fixed before any RHC fit):

* Candidate pool: ``POOL_SIZE`` points from a Latin hypercube with a fixed
  seed over (depth, width, log learning rate, log L2). The pool never depends
  on data, so a data-dependent choice among its members is a choice among a
  fixed finite set.
* Tuning tasks: H-V, Q-V, H-U, Q-U. Everything except the four searched
  coordinates (batch size, bandwidth multiplier, training objective, epochs,
  patience, scheduler) is taken from the matching S1 frozen config.
  NMMR-RW borrows the V configuration and NMMR-U+clip the U configuration
  (stage 1), so they are not tuned here.
* Inner split: ``nested_hpo_fold_split`` (3 folds stratified by A, 10% of
  each development part held out for early stopping). The RHC transform is
  fitted on each inner development part only.
* Screening: every candidate on every inner fold with one seed; all
  candidates on the same fold share the initialization and shuffle seeds
  (``fold_control_seeds``). Ranked by the size-weighted mean U score.
* Re-evaluation: the top ``TOP_K`` with ``N_REEVAL_SEEDS`` new seeds on the
  same inner folds; final score and its standard error from these units.
* Selection: ``pick_simplest_within_one_se`` (one-SE rule, simplest =
  shallower, narrower, stronger L2, shorter training), as in S1.

The same ``run_tuning_task`` serves stage 0 (the whole sample) and the
strict nested sensitivity analysis (an outer training fold, passed as a
sub-frame, so the outer evaluation rows are never seen).

Units are appended to ``units.csv`` as they finish; a rerun with the same
run config skips finished units, so an interrupted task can be resumed.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.ate.rhc_protocol_data import RHCFrameDataset, transformed_dataset
from src.models.experiment_common import (
    fold_control_seeds,
    fold_score_dispersion,
    insufficient_treatment_groups_reason,
    nested_hpo_fold_split,
    pick_simplest_within_one_se,
    weighted_fold_mean,
)
from src.models.hpo_protocol import FrozenConfig
from src.utils import empty_device_cache, get_device, make_seed_bundle, set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
S1_FROZEN_DIR = PROJECT_ROOT / "configs" / "sgd_frozen"
DEFAULT_POOL_PATH = PROJECT_ROOT / "configs" / "rhc" / "candidate_pool.json"

TUNING_TASKS: Tuple[str, ...] = ("H-V", "Q-V", "H-U", "Q-U")
S1_FROZEN_FILES: Dict[str, str] = {
    "H-V": "h_v_frozen_config.json", "Q-V": "q_v_frozen_config.json",
    "H-U": "h_u_frozen_config.json", "Q-U": "q_u_frozen_config.json",
}

POOL_SIZE = 20
POOL_SEED = 20261125
DEPTHS: Tuple[int, ...] = (2, 3, 4)
WIDTHS: Tuple[int, ...] = (32, 64, 128)
LR_RANGE: Tuple[float, float] = (1e-4, 5e-3)
L2_RANGE: Tuple[float, float] = (1e-7, 1e-3)

N_INNER_FOLDS = 3
EARLYSTOP_FRACTION = 0.1
TOP_K = 3
N_REEVAL_SEEDS = 3

UNIT_COLUMNS: List[str] = [
    "stage", "candidate_id", "seed", "fold", "init_seed", "shuffle_seed", "score_u", "score_v",
    "n_score", "train_seconds", "best_epoch", "stop_reason", "failure_reason",
]


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def make_candidate_pool(n: int = POOL_SIZE, seed: int = POOL_SEED) -> List[Dict[str, Any]]:
    """
    Latin hypercube over [0,1)^4, mapped to depth and width (each coordinate
    cut into equal thirds) and to log-uniform learning rate and L2. Depends
    only on (n, seed).
    """
    from scipy.stats import qmc

    u = qmc.LatinHypercube(d=4, seed=seed).random(n)

    def discrete(col: np.ndarray, levels: Sequence[int]) -> List[int]:
        return [int(levels[min(int(v * len(levels)), len(levels) - 1)]) for v in col]

    def log_uniform(col: np.ndarray, lo: float, hi: float) -> List[float]:
        return [float(math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo)))) for v in col]

    depth = discrete(u[:, 0], DEPTHS)
    width = discrete(u[:, 1], WIDTHS)
    lr = log_uniform(u[:, 2], *LR_RANGE)
    l2 = log_uniform(u[:, 3], *L2_RANGE)
    return [
        {"candidate_id": i, "network_depth": depth[i], "network_width": width[i],
         "learning_rate": lr[i], "l2_penalty": l2[i]}
        for i in range(n)
    ]


def pool_payload(pool: List[Dict[str, Any]], n: int = POOL_SIZE, seed: int = POOL_SEED) -> Dict[str, Any]:
    return {
        "generator": "scipy.stats.qmc.LatinHypercube",
        "pool_size": n, "pool_seed": seed,
        "search_space": {"network_depth": list(DEPTHS), "network_width": list(WIDTHS),
                         "learning_rate_log_uniform": list(LR_RANGE), "l2_penalty_log_uniform": list(L2_RANGE)},
        "candidates": pool,
    }


def save_candidate_pool(path: Path = DEFAULT_POOL_PATH) -> str:
    """Write the pool file (refuses to overwrite a different one); returns its SHA-256."""
    path = Path(path)
    text = json.dumps(pool_payload(make_candidate_pool()), indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"{path} exists with different content; refusing to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_candidate_pool(path: Path = DEFAULT_POOL_PATH) -> Tuple[List[Dict[str, Any]], str]:
    """Load the pool and check it equals what the fixed generator produces."""
    path = Path(path)
    text = path.read_text()
    payload = json.loads(text)
    expected = make_candidate_pool(payload["pool_size"], payload["pool_seed"])
    if payload["candidates"] != expected:
        raise ValueError(f"{path}: candidates differ from the fixed generator's output.")
    return payload["candidates"], hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------

def s1_base_train_params(task: str, frozen_dir: Path = S1_FROZEN_DIR) -> Dict[str, Any]:
    """Everything except the four searched coordinates, from the S1 frozen config."""
    if task not in TUNING_TASKS:
        raise ValueError(f"Unknown tuning task {task!r}; expected one of {TUNING_TASKS}.")
    frozen = FrozenConfig.load(Path(frozen_dir) / S1_FROZEN_FILES[task])
    if frozen.method_id != task:
        raise ValueError(f"S1 frozen config for {task} has method_id={frozen.method_id!r}.")
    return frozen.to_train_params(task.split("-")[0])


def candidate_train_params(base: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(base)
    for key in ("network_depth", "network_width", "learning_rate", "l2_penalty"):
        params[key] = candidate[key]
    return params


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def tuning_seeds(base_seed: int, task: str) -> Dict[str, Any]:
    """Inner split seed, screening seed and re-evaluation seeds of one task."""
    bundle = make_seed_bundle(base_seed, TUNING_TASKS.index(task))
    reeval = [make_seed_bundle(bundle.tuning_seed, k).init_seed for k in range(1, N_REEVAL_SEEDS + 1)]
    return {"split_seed": bundle.split_seed, "screen_seed": bundle.init_seed, "reeval_seeds": reeval}


# ---------------------------------------------------------------------------
# One training-and-scoring unit
# ---------------------------------------------------------------------------

def _scorer(task: str):
    if task.startswith("Q"):
        from src.models.NMMRq.nmmr_q_experiments import _compute_unified_q_scores, _create_trainer
        return _create_trainer, _compute_unified_q_scores
    from src.models.NMMRh.nmmr_h_experiments import _compute_unified_h_scores, _create_trainer
    return _create_trainer, _compute_unified_h_scores


def evaluate_unit(task: str, train_params: Dict[str, Any], frame: pd.DataFrame,
                  fold: Tuple[np.ndarray, np.ndarray, np.ndarray], fold_idx: int, seed: int,
                  split_seed: int, device: Any = None) -> Dict[str, Any]:
    """
    Train one candidate on one inner fold with one seed and score it on the
    fold's held-out part. The transform is fitted on the development part
    (gradient + early-stopping rows) only. Failures follow the S1 rules and
    give score_u = +inf.
    """
    device = device if device is not None else get_device()
    grad_idx, es_idx, score_idx = fold
    init_seed, shuffle_seed = fold_control_seeds(seed, fold_idx)
    row = {"fold": fold_idx, "init_seed": init_seed, "shuffle_seed": shuffle_seed,
           "score_u": float("inf"), "score_v": float("inf"), "n_score": len(score_idx),
           "train_seconds": float("nan"), "best_epoch": None, "stop_reason": None, "failure_reason": None}

    dataset = transformed_dataset(frame, np.concatenate([grad_idx, es_idx]), device=device)
    reason = insufficient_treatment_groups_reason(
        dataset, {"gradient_train": grad_idx, "earlystop_val": es_idx, "score": score_idx})
    if reason is not None:
        row["failure_reason"] = reason
        return row

    create_trainer, compute_scores = _scorer(task)
    set_random_seed(init_seed)
    trainer = create_trainer({}, train_params, split_seed, None, "rhc")
    batch_size = int(train_params["batch_size"])
    train_loader = DataLoader(Subset(dataset, grad_idx.tolist()), batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(shuffle_seed))
    es_loader = DataLoader(Subset(dataset, es_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)
    try:
        model, result = trainer.train(train_loader, es_loader)
        row.update(train_seconds=result.train_seconds, best_epoch=result.best_epoch, stop_reason=result.stop_reason)
        if result.stop_reason == "no_valid_checkpoint":
            row["failure_reason"] = "no_valid_checkpoint"
        elif result.n_nonfinite_train > 0 or result.n_nonfinite_val > 0:
            row["failure_reason"] = "nonfinite_during_training"
        else:
            score_v, score_u, n = compute_scores(trainer, model, dataset, score_idx.tolist())
            if not (math.isfinite(score_u) and math.isfinite(score_v)):
                row["failure_reason"] = "nonfinite_score"
            else:
                row.update(score_u=float(score_u), score_v=float(score_v), n_score=int(n))
    except Exception as exc:  # a failed unit is recorded, never dropped
        row["failure_reason"] = f"exception: {exc}"
    finally:
        del train_loader, es_loader
        empty_device_cache(device)
    return row


def summarize_candidate(units: pd.DataFrame, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Size-weighted mean U score over the units and its standard error (the
    `value`/`fold_se` keys `pick_simplest_within_one_se` reads). Any failed
    unit makes the candidate's value +inf, as in S1.
    """
    out = {k: candidate[k] for k in ("candidate_id", "network_depth", "network_width", "learning_rate", "l2_penalty")}
    out["n_units"] = int(len(units))
    out["total_train_seconds"] = float(np.nansum(units["train_seconds"].to_numpy(dtype=float)))
    failed = units["failure_reason"].notna() & (units["failure_reason"].astype(str) != "")
    if len(units) == 0 or failed.any():
        out.update(value=float("inf"), fold_se=0.0, fold_sd=0.0, value_v=float("inf"),
                   failure_reasons=";".join(units.loc[failed, "failure_reason"].astype(str)))
        return out
    scores = units["score_u"].astype(float).tolist()
    sizes = units["n_score"].astype(int).tolist()
    sd, se = fold_score_dispersion(scores)
    out.update(value=weighted_fold_mean(scores, sizes), fold_se=se, fold_sd=sd,
               value_v=weighted_fold_mean(units["score_v"].astype(float).tolist(), sizes), failure_reasons="")
    return out


# ---------------------------------------------------------------------------
# A whole tuning task
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def frame_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _load_units(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=UNIT_COLUMNS)
    return pd.read_csv(path, dtype={"failure_reason": object, "stop_reason": object})


def _append_unit(path: Path, row: Dict[str, Any]) -> None:
    pd.DataFrame([row], columns=UNIT_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_tuning_task(task: str, frame: pd.DataFrame, out_dir: Path, base_seed: int,
                    pool_path: Path = DEFAULT_POOL_PATH, frozen_dir: Path = S1_FROZEN_DIR,
                    device: Any = None, pool: Optional[List[Dict[str, Any]]] = None,
                    kernel_form: str = "joint", log=print) -> Dict[str, Any]:
    """
    Screen the pool, re-evaluate the top ``TOP_K``, apply the one-SE rule and
    write ``selected.json``. ``frame`` is the data the task may see (the whole
    sample, or one outer training fold). Resumable: finished units in
    ``units.csv`` are reused when the run config matches.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pool is None:
        pool, pool_sha = load_candidate_pool(pool_path)
    else:
        pool_sha = "in-memory"
    base = {**s1_base_train_params(task, frozen_dir), "kernel_form": kernel_form}
    seeds = tuning_seeds(base_seed, task)
    frame = frame.reset_index(drop=True)

    run_config = {
        "task": task, "base_seed": base_seed, "seeds": seeds, "pool_sha256": pool_sha,
        "n_rows": len(frame), "frame_sha256": frame_fingerprint(frame), "base_train_params": base,
        "n_inner_folds": N_INNER_FOLDS, "earlystop_fraction": EARLYSTOP_FRACTION,
        "top_k": TOP_K, "n_reeval_seeds": N_REEVAL_SEEDS, "ranking": "held-out U moment risk (reference kernel)",
        "selection": "pick_simplest_within_one_se over re-evaluation scores",
    }
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous.pop("git_commit", None)
        if previous != json.loads(json.dumps(run_config)):
            raise ValueError(f"{config_path} exists with a different run config; use a new output directory.")
    else:
        config_path.write_text(json.dumps({**run_config, "git_commit": _git_commit()}, indent=2))

    split_ds = RHCFrameDataset(frame)  # untransformed; only A is read for stratification
    folds = nested_hpo_fold_split(split_ds, N_INNER_FOLDS, seeds["split_seed"], EARLYSTOP_FRACTION)
    units_path = out_dir / "units.csv"
    done = _load_units(units_path)
    done_keys = {(r.stage, int(r.candidate_id), int(r.seed), int(r.fold)) for r in done.itertuples()}

    def run_units(stage: str, candidate: Dict[str, Any], seed_list: Sequence[int]) -> None:
        params = candidate_train_params(base, candidate)
        for seed in seed_list:
            for fold_idx, fold in enumerate(folds, start=1):
                key = (stage, candidate["candidate_id"], int(seed), fold_idx)
                if key in done_keys:
                    continue
                t0 = time.monotonic()
                row = evaluate_unit(task, params, frame, fold, fold_idx, int(seed), seeds["split_seed"], device)
                row.update(stage=stage, candidate_id=candidate["candidate_id"], seed=int(seed))
                _append_unit(units_path, row)
                done_keys.add(key)
                log(f"[{task} {stage}] cand {candidate['candidate_id']:2d} seed {seed} fold {fold_idx}: "
                    f"U={row['score_u']:.5g} ({time.monotonic() - t0:.0f}s){' FAIL ' + row['failure_reason'] if row['failure_reason'] else ''}")

    for candidate in pool:
        run_units("screen", candidate, [seeds["screen_seed"]])
    units = _load_units(units_path)
    screen = pd.DataFrame([
        summarize_candidate(units[(units.stage == "screen") & (units.candidate_id == c["candidate_id"])], c)
        for c in pool
    ]).sort_values(["value", "candidate_id"]).reset_index(drop=True)
    screen.to_csv(out_dir / "screen_summary.csv", index=False)
    finite = screen[np.isfinite(screen["value"])]
    if finite.empty:
        raise RuntimeError(f"{task}: every candidate failed in screening.")
    top = finite.head(TOP_K)
    by_id = {c["candidate_id"]: c for c in pool}

    for cid in top["candidate_id"]:
        run_units("reeval", by_id[int(cid)], seeds["reeval_seeds"])
    units = _load_units(units_path)
    finalists = [
        summarize_candidate(units[(units.stage == "reeval") & (units.candidate_id == int(cid))], by_id[int(cid)])
        for cid in top["candidate_id"]
    ]
    pd.DataFrame(finalists).to_csv(out_dir / "reeval_summary.csv", index=False)
    valid = [c for c in finalists if math.isfinite(c["value"])]
    if not valid:
        raise RuntimeError(f"{task}: every finalist failed in re-evaluation.")
    selected, pure_min = pick_simplest_within_one_se(valid)

    payload = {
        "task": task, "selected_candidate_id": int(selected["candidate_id"]),
        "pure_min_candidate_id": int(pure_min["candidate_id"]),
        "train_params": candidate_train_params(base, by_id[int(selected["candidate_id"])]),
        "selected_score_u": selected["value"], "selected_se": selected["fold_se"],
        "pure_min_score_u": pure_min["value"], "pure_min_se": pure_min["fold_se"],
        "run_config": run_config, "git_commit": _git_commit(),
    }
    (out_dir / "selected.json").write_text(json.dumps(payload, indent=2))
    log(f"[{task}] selected candidate {payload['selected_candidate_id']} "
        f"(pure minimum {payload['pure_min_candidate_id']})")
    return payload


def load_selected(path: Path, expected_task: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("task") != expected_task:
        raise ValueError(f"{path}: task={payload.get('task')!r}, expected {expected_task!r}.")
    return payload
