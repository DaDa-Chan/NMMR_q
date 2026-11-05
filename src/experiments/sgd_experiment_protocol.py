"""
P0/E1/E2 shared SGD protocol (plan/p0_e1_e2/01_shared_protocol.md,
plan/p0_e1_e2/02_methods.md). This module owns everything P0/E1/E2 have in
common: scenario dispatch, repetition seeds, the A-stratified fold plan
shared by every method, method dispatch (plain functions, no class
hierarchy -- "oracle"/"parametric_proximal" need no model object at all,
"nmmr_u"/"nmmr_v" dispatch against real FrozenConfigs), the pure
effect/bridge-error computation, and writing the minimal result-file
structure below. It never tunes hyperparameters -- Optuna runs exclusively
in scripts/run_pilot_protocol.py. This round adds "oracle" (plan/p0_e1_e2/
02_methods.md section 2: the S1 scenario registry's own analytical true
bridges, no fitting at all) and "parametric_proximal" (section 3: the
correctly-specified linear H / exponential-tilt Q proximal model, fit
per-fold via the closed-form/moment-equation solvers in
src/models/LINEAR/linear_model.py). LS/LS-QF/2SLS/RKHS/S2-S4 remain out of
scope.

Output layout (plan/p0_e1_e2/01_shared_protocol.md section 7, this round's
minimal subset only -- *_summary.csv and plots are later-phase work):

    results/<run_id>/
    ├── run_config.json
    ├── common/
    │   ├── repetitions.csv
    │   ├── folds.csv
    │   ├── fold_diagnostics.csv   (per-fold H/Q training diagnostics)
    │   └── failures.csv
    ├── e1/
    │   └── bridge_results.csv
    └── e2/
        └── effect_results.csv     (also carries h/q_train_seconds,
                                     evaluation_seconds, total_seconds)

No resume/checkpoint support: `ensure_run_config` refuses to run into an
`output_dir` that already has a `run_config.json`, so every real run must
use a brand-new `output_dir`. If a run is interrupted or fails outright,
delete that incomplete output_dir and start over from scratch (see
README's "SGD P0/E1/E2 实验运行规则") -- this module never reads back its
own previous CSV rows to decide what to skip or repair. A method's result
is only ever written once BOTH its bridge and effect metrics, and every
required value (OOF/test predictions, POR/PIPW/PDR, bridge MSE, held-out
moment risk), are finite -- NonFiniteResultError routes it to
common/failures.csv instead, and that one method's failure does not stop
the rest of the run.

RHC's existing double_rob.py path is untouched and does not go through this
module at all.
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.ate.data_class import SGDDataset
from src.models.experiment_common import _get_a_labels
from src.models.hpo_protocol import (
    FrozenConfig,
    _dataframe_sha256,
    _get_git_commit,
    validate_frozen_config,
    validate_method_ids,
)
from src.utils import get_device, make_seed_bundle

from .effects import compute_bridge_mae, compute_bridge_mse, compute_effect_scores, compute_q_distribution_diagnostics
from .scenarios import ScenarioSpec, get_scenario

# This round's explicit method scope: oracle (plan/p0_e1_e2/02_methods.md
# section 2), parametric_proximal (section 3), and nmmr_u/nmmr_v (section 4,
# pairing H-U+Q-U / H-V+Q-V). LS/LS-QF/2SLS/RKHS/S2-S4 remain out of scope.
ALLOWED_METHODS: Tuple[str, ...] = ("oracle", "parametric_proximal", "nmmr_u", "nmmr_v")
# Only these methods read a FrozenConfig -- oracle uses the scenario
# registry's analytical truth, parametric_proximal only ever uses the
# correct functional FORM (parameters are estimated per fold, never frozen).
METHODS_REQUIRING_FROZEN_CONFIG: Tuple[str, ...] = ("nmmr_u", "nmmr_v")
METHOD_FROZEN_IDS: Dict[str, Tuple[str, str]] = {
    "nmmr_u": ("H-U", "Q-U"),
    "nmmr_v": ("H-V", "Q-V"),
}

# Stable integer tags so a (scenario, method, bridge) combination always
# derives an independent seed, regardless of call order (plan/p0_e1_e2/
# 01_shared_protocol.md section 5: "每折初始化从 (method,repetition,fold,
# init_seed) 派生"); scenario/repetition are already baked into the
# RepetitionSeeds this is derived from.
_SCENARIO_ORDINALS: Dict[str, int] = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
_METHOD_TAGS: Dict[str, int] = {"nmmr_u": 101, "nmmr_v": 102}
_BRIDGE_TAGS: Dict[str, int] = {"H": 1, "Q": 0}

RESULT_ROW_KEY_COLUMNS: List[str] = ["run_stage", "scenario", "n", "repetition", "method", "variant"]


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepetitionSeeds:
    """The four independently-saved seeds one (scenario, n, repetition)
    needs (plan/p0_e1_e2/00_p0_correctness.md section 3 / plan/实验计划.md
    section 3.2)."""
    data_seed: int
    test_seed: int
    split_seed: int
    init_seed: int


def derive_repetition_seeds(base_seed: int, scenario_id: str, n: int, repetition: int) -> RepetitionSeeds:
    """
    Deterministic, independent seeds for one (scenario, n, repetition),
    using the project's existing SeedSequence-based derivation
    (src/utils.make_seed_bundle) rather than inventing a new one.
    `test_seed` is not one of SeedBundle's five roles, so it is taken from a
    SEPARATE spawn (extra trailing entropy component `1`) rather than
    reusing any of data/split/init/tuning/bootstrap for a second purpose.
    """
    if scenario_id not in _SCENARIO_ORDINALS:
        raise ValueError(f"Unknown scenario_id {scenario_id!r}; known: {sorted(_SCENARIO_ORDINALS)}.")
    scenario_ord = _SCENARIO_ORDINALS[scenario_id]
    bundle = make_seed_bundle(base_seed, scenario_ord, n, repetition)
    test_bundle = make_seed_bundle(base_seed, scenario_ord, n, repetition, 1)
    return RepetitionSeeds(
        data_seed=bundle.data_seed,
        test_seed=test_bundle.data_seed,
        split_seed=bundle.split_seed,
        init_seed=bundle.init_seed,
    )


def _derive_bridge_init_seed(base_init_seed: int, method: str, bridge: str) -> int:
    """Base seed each fold's model init derives from, for ONE (method,
    bridge) pair within one repetition -- independent across method AND
    bridge even when both share the same `base_init_seed`."""
    if method not in _METHOD_TAGS:
        raise ValueError(f"Unknown method {method!r}; known: {sorted(_METHOD_TAGS)}.")
    if bridge not in _BRIDGE_TAGS:
        raise ValueError(f"Unknown bridge {bridge!r}; expected 'H' or 'Q'.")
    return make_seed_bundle(base_init_seed, _METHOD_TAGS[method], _BRIDGE_TAGS[bridge]).init_seed


# ---------------------------------------------------------------------------
# Fold plan: one A-stratified split, shared by H and Q, used as-is
# ---------------------------------------------------------------------------

def _hash_indices(idx: Sequence[int]) -> str:
    arr = np.asarray(sorted(int(i) for i in idx), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class FoldRecord:
    fold_id: int
    n_train: int
    n_eval: int
    n_train_a0: int
    n_train_a1: int
    n_eval_a0: int
    n_eval_a1: int
    train_idx_hash: str
    eval_idx_hash: str


@dataclass(frozen=True)
class FoldPlan:
    n_splits: int
    split_seed: int
    splits: List[Tuple[np.ndarray, np.ndarray]]
    records: List[FoldRecord]
    fold_plan_hash: str


def build_fold_plan(dataset: Any, n_splits: int, split_seed: int) -> FoldPlan:
    """
    The ONE A-stratified fold plan a repetition uses for both H and Q
    (plan/p0_e1_e2/01_shared_protocol.md section 4). Every check here is
    explicit -- this never relies on StratifiedKFold's own warnings or
    implicit fallback behavior (Phase 1.1 item 1):

    1. `A` must contain both 0 and 1, and no other value.
    2. Each treatment group must have at least `n_splits` samples.
    3. Each fold's train AND eval split must itself contain both groups.

    Raises a `ValueError` naming the specific violation on any failure --
    never falls back to a plain KFold; the caller records that as this
    repetition's failure instead.
    """
    indices = np.arange(len(dataset))
    a_labels = _get_a_labels(dataset, indices)

    valid_a = np.isin(a_labels, [0.0, 1.0])
    if not valid_a.all():
        bad = sorted(set(a_labels[~valid_a].tolist()))
        raise ValueError(f"A must only take values in {{0, 1}}; found other value(s): {bad}.")
    n_a0 = int((a_labels < 0.5).sum())
    n_a1 = int((a_labels > 0.5).sum())
    if n_a0 == 0 or n_a1 == 0:
        raise ValueError(
            f"A must contain both treatment groups 0 and 1; found only one group "
            f"(n_a0={n_a0}, n_a1={n_a1})."
        )
    if n_a0 < n_splits or n_a1 < n_splits:
        raise ValueError(
            f"Each treatment group must have at least n_splits={n_splits} samples to "
            f"stratify into {n_splits} folds; got n_a0={n_a0}, n_a1={n_a1}."
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    splits = list(splitter.split(indices, a_labels))

    records: List[FoldRecord] = []
    for fold_id, (train_idx, eval_idx) in enumerate(splits, start=1):
        train_a = a_labels[train_idx]
        eval_a = a_labels[eval_idx]
        n_train_a0 = int((train_a < 0.5).sum())
        n_train_a1 = int((train_a > 0.5).sum())
        n_eval_a0 = int((eval_a < 0.5).sum())
        n_eval_a1 = int((eval_a > 0.5).sum())
        if n_train_a0 == 0 or n_train_a1 == 0:
            raise ValueError(
                f"Fold {fold_id}'s train split is missing a treatment group "
                f"(n_train_a0={n_train_a0}, n_train_a1={n_train_a1})."
            )
        if n_eval_a0 == 0 or n_eval_a1 == 0:
            raise ValueError(
                f"Fold {fold_id}'s eval split is missing a treatment group "
                f"(n_eval_a0={n_eval_a0}, n_eval_a1={n_eval_a1})."
            )
        records.append(FoldRecord(
            fold_id=fold_id,
            n_train=len(train_idx), n_eval=len(eval_idx),
            n_train_a0=n_train_a0, n_train_a1=n_train_a1,
            n_eval_a0=n_eval_a0, n_eval_a1=n_eval_a1,
            train_idx_hash=_hash_indices(train_idx), eval_idx_hash=_hash_indices(eval_idx),
        ))
    fold_plan_hash = hashlib.sha256(
        "|".join(r.train_idx_hash + r.eval_idx_hash for r in records).encode()
    ).hexdigest()
    return FoldPlan(n_splits=n_splits, split_seed=split_seed, splits=splits, records=records,
                     fold_plan_hash=fold_plan_hash)


# ---------------------------------------------------------------------------
# FrozenConfig loading (reuses hpo_protocol.FrozenConfig entirely -- never
# re-implements its field layout; mirrors src/experiment.py's
# load_frozen_config_for_standalone without importing from it, since this
# round must not modify or depend on src/experiment.py)
# ---------------------------------------------------------------------------

def load_frozen_config(path: Path, expected_method_id: str, expected_n_splits: int) -> Tuple[FrozenConfig, Dict[str, Any]]:
    """
    Load and validate one FrozenConfig for the formal SGD NMMR path
    (plan/p0_e1_e2/01_shared_protocol.md section 2): `confirm_status==
    "passed"`, method id matches what the caller expects, no leftover
    search space, and `n_splits` matches this protocol run's own n_splits.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen config not found: {path}")
    frozen = FrozenConfig.load(path)
    validate_method_ids([frozen.method_id])
    if frozen.method_id != expected_method_id:
        raise ValueError(
            f"{path}: method_id={frozen.method_id!r} does not match the expected "
            f"{expected_method_id!r}."
        )
    if frozen.confirm_status != "passed":
        raise ValueError(
            f"{path} (method_id={frozen.method_id}): confirm_status={frozen.confirm_status!r}; "
            "only a confirm_status=='passed' frozen config may drive the formal SGD protocol."
        )
    if frozen.n_splits != expected_n_splits:
        raise ValueError(
            f"{path}: frozen config n_splits={frozen.n_splits} does not match this "
            f"protocol run's n_splits={expected_n_splits}."
        )
    model_kind = frozen.method_id.split("-")[0]
    train_params = frozen.to_train_params(model_kind)
    validate_frozen_config(train_params)
    return frozen, train_params


# ---------------------------------------------------------------------------
# Oracle: the scenario registry's own analytical true bridges, no fitting
# (plan/p0_e1_e2/02_methods.md section 2).
# ---------------------------------------------------------------------------

def run_oracle_method(
    train_dataset: Any,
    test_dataset: Any,
    scenario: ScenarioSpec,
) -> Dict[str, Any]:
    """
    No model, no cross-fitting loop, no frozen config: every prediction is
    just `scenario.true_h`/`scenario.true_q` evaluated at the actual
    (W/Z/X) of each sample, at whatever treatment level is asked for. The
    OOF arrays use each sample's own observed A (not estimated), since
    there is nothing to hold out -- the true bridge is the same function
    regardless of which fold a sample would have been in.
    """
    train_w = train_dataset.W.detach().cpu().numpy().reshape(-1)
    train_z = train_dataset.Z.detach().cpu().numpy().reshape(-1)
    train_x = train_dataset.X.detach().cpu().numpy()
    train_a = train_dataset.A.detach().cpu().numpy().reshape(-1)
    train_ones = np.ones_like(train_a)
    train_zeros = np.zeros_like(train_a)

    n_test = len(test_dataset)
    test_w = test_dataset.W.detach().cpu().numpy().reshape(-1)
    test_z = test_dataset.Z.detach().cpu().numpy().reshape(-1)
    test_x = test_dataset.X.detach().cpu().numpy()
    test_ones = np.ones(n_test)
    test_zeros = np.zeros(n_test)

    return {
        "h_observed_oof": scenario.true_h(train_w, train_a, train_x),
        "h0_oof": scenario.true_h(train_w, train_zeros, train_x),
        "h1_oof": scenario.true_h(train_w, train_ones, train_x),
        "q_observed_oof": scenario.true_q(train_z, train_a, train_x),
        "h0_test": scenario.true_h(test_w, test_zeros, test_x),
        "h1_test": scenario.true_h(test_w, test_ones, test_x),
        "q0_test": scenario.true_q(test_z, test_zeros, test_x),
        "q1_test": scenario.true_q(test_z, test_ones, test_x),
        "fold_diagnostics_h": [],
        "fold_diagnostics_q": [],
    }


# ---------------------------------------------------------------------------
# Correctly-specified parametric proximal model: correct functional FORM,
# parameters estimated per outer fold (plan/p0_e1_e2/02_methods.md section
# 3). Reuses src/models/LINEAR/linear_model.py's ProximalLinearModel
# entirely -- never re-implements its moment equations here.
# ---------------------------------------------------------------------------

def run_parametric_proximal_method(
    train_dataset: Any,
    test_dataset: Any,
    fold_plan: FoldPlan,
) -> Dict[str, Any]:
    """
    For each outer fold in `fold_plan` (the SAME plan NMMR uses): fit H/Q on
    that fold's train rows only, predict its eval rows (OOF) and the
    independent test set, then average the test-set predictions across
    folds with equal weight (same convention as NMMR's cross-fitting). A
    solver that does not converge is a method failure (raised here, caught
    by the caller) -- never silently accepted with a bad fit.
    """
    from src.models.LINEAR.linear_model import ProximalLinearModel

    train_df = train_dataset.data
    test_df = test_dataset.data
    n_train = len(train_dataset)
    n_test = len(test_dataset)

    h_observed_oof = np.full(n_train, np.nan)
    h0_oof = np.full(n_train, np.nan)
    h1_oof = np.full(n_train, np.nan)
    q_observed_oof = np.full(n_train, np.nan)

    test_W = test_df[['W']].values
    test_Z = test_df[['Z']].values
    test_X = test_df[['X1', 'X2']].values
    test_ones_a = np.ones(n_test)
    test_zeros_a = np.zeros(n_test)
    test_h0_per_fold: List[np.ndarray] = []
    test_h1_per_fold: List[np.ndarray] = []
    test_q0_per_fold: List[np.ndarray] = []
    test_q1_per_fold: List[np.ndarray] = []

    fold_diagnostics_h: List[Dict[str, Any]] = []
    fold_diagnostics_q: List[Dict[str, Any]] = []

    for fold_id, (train_idx, eval_idx) in enumerate(fold_plan.splits, start=1):
        fold_train_df = train_df.iloc[train_idx]
        fold_eval_df = train_df.iloc[eval_idx]
        model = ProximalLinearModel(fold_train_df)

        h_start = time.monotonic()
        b_hat, used_lstsq = model.solve_h()
        h_seconds = time.monotonic() - h_start

        q_start = time.monotonic()
        t_hat, q_converged, q_message, q_residual_norm = model.solve_q()
        q_seconds = time.monotonic() - q_start
        if not q_converged:
            raise RuntimeError(
                f"Fold {fold_id}'s Q moment-equation solver did not converge: {q_message}"
            )

        eval_W = fold_eval_df[['W']].values
        eval_Z = fold_eval_df[['Z']].values
        eval_X = fold_eval_df[['X1', 'X2']].values
        eval_A = fold_eval_df['A'].values
        eval_ones_a = np.ones(len(eval_idx))
        eval_zeros_a = np.zeros(len(eval_idx))

        h_observed_oof[eval_idx] = model.h_func(b_hat, eval_W, eval_A, eval_X)
        h0_oof[eval_idx] = model.h_func(b_hat, eval_W, eval_zeros_a, eval_X)
        h1_oof[eval_idx] = model.h_func(b_hat, eval_W, eval_ones_a, eval_X)
        q_observed_oof[eval_idx] = model.q_func(t_hat, eval_Z, eval_A, eval_X)

        test_h0_per_fold.append(model.h_func(b_hat, test_W, test_zeros_a, test_X))
        test_h1_per_fold.append(model.h_func(b_hat, test_W, test_ones_a, test_X))
        test_q0_per_fold.append(model.q_func(t_hat, test_Z, test_zeros_a, test_X))
        test_q1_per_fold.append(model.q_func(t_hat, test_Z, test_ones_a, test_X))

        fold_diagnostics_h.append({
            "fold_id": fold_id,
            "solve_success": bool(np.all(np.isfinite(b_hat))),
            "train_seconds": h_seconds,
            "used_lstsq": bool(used_lstsq),
            "moment_residual_norm": None,
        })
        fold_diagnostics_q.append({
            "fold_id": fold_id,
            "solve_success": bool(q_converged and np.all(np.isfinite(t_hat))),
            "train_seconds": q_seconds,
            "used_lstsq": None,
            "moment_residual_norm": q_residual_norm,
        })

    return {
        "h_observed_oof": h_observed_oof,
        "h0_oof": h0_oof,
        "h1_oof": h1_oof,
        "q_observed_oof": q_observed_oof,
        "h0_test": np.mean(np.stack(test_h0_per_fold, axis=0), axis=0),
        "h1_test": np.mean(np.stack(test_h1_per_fold, axis=0), axis=0),
        "q0_test": np.mean(np.stack(test_q0_per_fold, axis=0), axis=0),
        "q1_test": np.mean(np.stack(test_q1_per_fold, axis=0), axis=0),
        "fold_diagnostics_h": fold_diagnostics_h,
        "fold_diagnostics_q": fold_diagnostics_q,
    }


# ---------------------------------------------------------------------------
# NMMR-U/V dispatch
# ---------------------------------------------------------------------------

def run_nmmr_method(
    method: str,
    train_dataset: Any,
    test_dataset: Any,
    fold_plan: FoldPlan,
    rep_seeds: RepetitionSeeds,
    h_train_params: Dict[str, Any],
    q_train_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fit H and Q (NMMR-U or NMMR-V, picked by which train_params are passed
    in) on the SAME externally-built `fold_plan`, and return the unified
    prediction set NEXT_IMPLEMENTATION_PROMPT.md section 5 asks for.
    """
    from src.models.NMMRh.nmmr_h_experiments import _run_cross_fitting as run_cf_h
    from src.models.NMMRq.nmmr_q_experiments import _run_cross_fitting as run_cf_q

    q_init_seed = _derive_bridge_init_seed(rep_seeds.init_seed, method, "Q")
    h_init_seed = _derive_bridge_init_seed(rep_seeds.init_seed, method, "H")

    _, _, q_oof, q_fold_diag, q_test = run_cf_q(
        data_name="sgd", dataset=train_dataset, n_splits=fold_plan.n_splits,
        train_params=q_train_params, data_configs={}, random_seed=rep_seeds.split_seed,
        log_folder=None, tag=f"{method}_q", fold_plan=fold_plan.splits,
        init_seed=q_init_seed, test_dataset=test_dataset,
    )
    _, _, h_oof, h_fold_diag, h_test = run_cf_h(
        data_name="sgd", dataset=train_dataset, n_splits=fold_plan.n_splits,
        train_params=h_train_params, data_configs={}, random_seed=rep_seeds.split_seed,
        log_folder=None, tag=f"{method}_h", fold_plan=fold_plan.splits,
        init_seed=h_init_seed, test_dataset=test_dataset,
    )
    return {
        "q_observed_oof": q_oof.reshape(-1),
        "h_observed_oof": h_oof["h_fact"].reshape(-1),
        "h0_oof": h_oof["h_0"].reshape(-1),
        "h1_oof": h_oof["h_1"].reshape(-1),
        "q0_test": q_test["q0_test"],
        "q1_test": q_test["q1_test"],
        "h0_test": h_test["h0_test"],
        "h1_test": h_test["h1_test"],
        "fold_diagnostics_h": h_fold_diag,
        "fold_diagnostics_q": q_fold_diag,
    }


class NonFiniteResultError(RuntimeError):
    """Raised when one of a method's REQUIRED results (OOF/test bridge
    predictions, POR/PIPW/PDR, bridge MSE, held-out moment risk) contains
    NaN/Inf (Phase 1.1 item 5). Only finiteness is checked here -- never a
    distance-from-truth threshold, a runtime budget, or whether training
    used every available epoch."""


def _require_finite(name: str, values: Any) -> None:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise NonFiniteResultError(f"{name} contains NaN/Inf.")


# Union schema for common/fold_diagnostics.csv: NMMR rows and
# parametric_proximal rows report different diagnostics (training-loop vs.
# solver), and oracle reports none at all -- every row still carries every
# column (unused ones left as None/empty) so appending rows from different
# methods into the SAME csv never misaligns columns.
_FOLD_DIAG_COLUMNS: Tuple[str, ...] = (
    "method_id", "bridge_type", "fold_id",
    "best_epoch", "stop_reason", "held_out_moment_risk", "train_seconds", "best_state_restored",
    "solve_success", "used_lstsq", "moment_residual_norm",
)


def _make_fold_diag_row(method_id: str, bridge_type: str, fold_id: int, **kwargs: Any) -> Dict[str, Any]:
    unknown = set(kwargs) - set(_FOLD_DIAG_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown fold diagnostic column(s): {sorted(unknown)}.")
    row = {col: None for col in _FOLD_DIAG_COLUMNS}
    row.update(method_id=method_id, bridge_type=bridge_type, fold_id=fold_id, **kwargs)
    return row


def compute_method_results(
    method: str,
    train_dataset: Any,
    test_dataset: Any,
    fold_plan: FoldPlan,
    rep_seeds: RepetitionSeeds,
    h_train_params: Dict[str, Any],
    q_train_params: Dict[str, Any],
    scenario: ScenarioSpec,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    One method's bridge-error row (E1), effect row (E2), and per-fold
    training-diagnostic rows for one repetition, computed from a single H/Q
    fit (plan/p0_e1_e2/plan.md section 2: "一次拟合，同时服务 E1 与 E2").
    `h_train_params`/`q_train_params` are only used (and only required) for
    "nmmr_u"/"nmmr_v"; `oracle` and `parametric_proximal` ignore them (no
    frozen config involved at all).

    Raises `NonFiniteResultError` (Phase 1.1 item 5) if any required result
    -- OOF/test predictions, POR/PIPW/PDR, bridge MSE, (NMMR only) held-out
    moment risk -- is not finite; the caller records that as a failure
    instead of writing a row. A parametric_proximal fold whose Q solver
    does not converge raises in `run_parametric_proximal_method` itself,
    caught the same way. h_train_seconds/q_train_seconds (summed from each
    bridge's own fold_diagnostics -- 0 for oracle, which has none) and
    evaluation_seconds (everything else) are purely diagnostic and never
    gate success.
    """
    t_start = time.monotonic()
    if method == "oracle":
        preds = run_oracle_method(train_dataset, test_dataset, scenario)
    elif method == "parametric_proximal":
        preds = run_parametric_proximal_method(train_dataset, test_dataset, fold_plan)
    elif method in ("nmmr_u", "nmmr_v"):
        preds = run_nmmr_method(
            method, train_dataset, test_dataset, fold_plan, rep_seeds, h_train_params, q_train_params,
        )
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {ALLOWED_METHODS}.")

    _require_finite("q_observed_oof", preds["q_observed_oof"])
    _require_finite("h_observed_oof", preds["h_observed_oof"])
    _require_finite("h0_oof", preds["h0_oof"])
    _require_finite("h1_oof", preds["h1_oof"])
    _require_finite("h0_test", preds["h0_test"])
    _require_finite("h1_test", preds["h1_test"])
    _require_finite("q0_test", preds["q0_test"])
    _require_finite("q1_test", preds["q1_test"])
    if method in ("nmmr_u", "nmmr_v"):
        # held-out moment risk is a training diagnostic specific to NMMR's
        # own fold_diagnostics shape; parametric_proximal/oracle don't have it.
        for d in preds["fold_diagnostics_h"] + preds["fold_diagnostics_q"]:
            _require_finite(f"held_out_moment_risk (fold {d['fold']})", d["moment_risk"])

    y = train_dataset.Y.detach().cpu().numpy().reshape(-1)
    a = train_dataset.A.detach().cpu().numpy().reshape(-1)
    effect = compute_effect_scores(
        y=y, a=a, q_observed=preds["q_observed_oof"], h_observed=preds["h_observed_oof"],
        h1=preds["h1_oof"], h0=preds["h0_oof"],
    )
    _require_finite("tau_por", effect["tau_por"])
    _require_finite("tau_pipw", effect["tau_pipw"])
    _require_finite("tau_pdr", effect["tau_pdr"])

    n_test = len(test_dataset)
    test_w = test_dataset.W.detach().cpu().numpy().reshape(-1)
    test_z = test_dataset.Z.detach().cpu().numpy().reshape(-1)
    test_x = test_dataset.X.detach().cpu().numpy()
    zeros = np.zeros(n_test)
    ones = np.ones(n_test)
    true_h0 = scenario.true_h(test_w, zeros, test_x)
    true_h1 = scenario.true_h(test_w, ones, test_x)
    true_q0 = scenario.true_q(test_z, zeros, test_x)
    true_q1 = scenario.true_q(test_z, ones, test_x)

    h_mse_a0 = compute_bridge_mse(preds["h0_test"], true_h0)
    h_mse_a1 = compute_bridge_mse(preds["h1_test"], true_h1)
    q_mse_a0 = compute_bridge_mse(preds["q0_test"], true_q0)
    q_mse_a1 = compute_bridge_mse(preds["q1_test"], true_q1)
    _require_finite("h_mse", [h_mse_a0, h_mse_a1])
    _require_finite("q_mse", [q_mse_a0, q_mse_a1])

    h_mae_a0 = compute_bridge_mae(preds["h0_test"], true_h0)
    h_mae_a1 = compute_bridge_mae(preds["h1_test"], true_h1)
    q_mae_a0 = compute_bridge_mae(preds["q0_test"], true_q0)
    q_mae_a1 = compute_bridge_mae(preds["q1_test"], true_q1)

    q_diag = compute_q_distribution_diagnostics(np.concatenate([preds["q0_test"], preds["q1_test"]]))
    t_eval_done = time.monotonic()

    h_train_seconds = float(sum(d["train_seconds"] for d in preds["fold_diagnostics_h"]))
    q_train_seconds = float(sum(d["train_seconds"] for d in preds["fold_diagnostics_q"]))
    total_seconds = t_eval_done - t_start
    # Everything NOT spent inside the fold loop's own fit step (OOF/test
    # prediction, plus the effect/bridge-error combination here) -- neither
    # _run_cross_fitting nor run_parametric_proximal_method separately times
    # its post-fit prediction step, so this is a simple diagnostic
    # approximation, not a precise breakdown. h_train_seconds/q_train_seconds
    # are always 0 for oracle (no fit at all, see run_oracle_method), but
    # oracle still spends nonzero time evaluating true_h/true_q and the
    # effect/bridge-error formulas, so evaluation_seconds/total_seconds are
    # NOT 0 for it.
    evaluation_seconds = max(0.0, total_seconds - h_train_seconds - q_train_seconds)

    bridge_row = {
        "h_mse_a0": h_mse_a0, "h_mse_a1": h_mse_a1, "h_mse": (h_mse_a0 + h_mse_a1) / 2.0,
        "h_mae_a0": h_mae_a0, "h_mae_a1": h_mae_a1, "h_mae": (h_mae_a0 + h_mae_a1) / 2.0,
        "q_mse_a0": q_mse_a0, "q_mse_a1": q_mse_a1, "q_mse": (q_mse_a0 + q_mse_a1) / 2.0,
        "q_mae_a0": q_mae_a0, "q_mae_a1": q_mae_a1, "q_mae": (q_mae_a0 + q_mae_a1) / 2.0,
        **q_diag,
    }
    effect_row = {
        "tau_por": effect["tau_por"], "tau_pipw": effect["tau_pipw"], "tau_pdr": effect["tau_pdr"],
        "h_train_seconds": h_train_seconds, "q_train_seconds": q_train_seconds,
        "evaluation_seconds": evaluation_seconds, "total_seconds": total_seconds,
    }

    fold_diag_rows: List[Dict[str, Any]] = []
    if method in ("nmmr_u", "nmmr_v"):
        for bridge_type, diags in (("H", preds["fold_diagnostics_h"]), ("Q", preds["fold_diagnostics_q"])):
            for d in diags:
                fold_diag_rows.append(_make_fold_diag_row(
                    method_id=method, bridge_type=bridge_type, fold_id=d["fold"],
                    best_epoch=d["best_epoch"], stop_reason=d["stop_reason"],
                    held_out_moment_risk=d["moment_risk"], train_seconds=d["train_seconds"],
                    best_state_restored=d["best_state_restored"],
                ))
    elif method == "parametric_proximal":
        for bridge_type, diags in (("H", preds["fold_diagnostics_h"]), ("Q", preds["fold_diagnostics_q"])):
            for d in diags:
                fold_diag_rows.append(_make_fold_diag_row(
                    method_id=method, bridge_type=bridge_type, fold_id=d["fold_id"],
                    solve_success=d["solve_success"], train_seconds=d["train_seconds"],
                    used_lstsq=d["used_lstsq"], moment_residual_norm=d["moment_residual_norm"],
                ))
    # oracle: no fold diagnostics at all (plan asks that it not write fake
    # best_epoch/stop_reason/training records).

    return bridge_row, effect_row, fold_diag_rows


# ---------------------------------------------------------------------------
# Result files: run_config.json + the five CSVs, append-only. No resume
# support (see run_sgd_protocol's docstring) -- every real run starts from
# a brand-new, empty output_dir.
# ---------------------------------------------------------------------------

def build_run_config(
    run_stage: str,
    scenario_ids: Sequence[str],
    methods: Sequence[str],
    sample_sizes: Sequence[int],
    repetitions: int,
    test_size: int,
    n_splits: int,
    base_seed: int,
    frozen_config_hashes: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "run_stage": run_stage,
        "scenarios": list(scenario_ids),
        "methods": list(methods),
        "sample_sizes": list(sample_sizes),
        "repetitions": repetitions,
        "test_size": test_size,
        "n_splits": n_splits,
        "base_seed": base_seed,
        "frozen_config_hashes": dict(sorted(frozen_config_hashes.items())),
        "software_version": _get_git_commit(),
    }


def ensure_run_config(output_dir: Path, run_config: Dict[str, Any]) -> None:
    """
    Writes run_config.json -- and REFUSES to run at all if `output_dir`
    already has one (no resume/checkpoint support, see this module's
    docstring: every real run must use a brand-new --output_dir; delete an
    incomplete one and start over rather than pointing a new run at it).
    """
    path = Path(output_dir) / "run_config.json"
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Each run must use a brand-new --output_dir; if a "
            "previous run into this directory was interrupted or failed, delete the whole "
            "directory and run again from scratch instead of pointing another run at it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_config, indent=2, sort_keys=True, default=str))


def _append_rows(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_sgd_protocol(
    run_stage: str,
    scenario_ids: Sequence[str],
    methods: Sequence[str],
    sample_sizes: Sequence[int],
    repetitions: int,
    test_size: int,
    n_splits: int,
    base_seed: int,
    frozen_config_paths: Dict[str, Path],
    output_dir: Path,
) -> Dict[str, int]:
    """
    Run P0/E1/E2's shared SGD protocol for every (scenario, n, repetition,
    method) combination, writing the minimal result structure under
    `output_dir`. One method's failure is recorded in common/failures.csv
    and does not stop the rest of that repetition or any other repetition
    (NEXT_IMPLEMENTATION_PROMPT.md 测试与验收 item 9 / 00_p0_correctness.md
    验收标准 item 9). Requesting a disabled scenario, or a method outside
    `ALLOWED_METHODS`, fails immediately and loudly instead.

    No resume support: `ensure_run_config` requires a brand-new
    `output_dir` (see this module's docstring) -- every (scenario, n,
    repetition, method) combination requested is always run, and every row
    is always freshly appended.
    """
    unknown_methods = [m for m in methods if m not in ALLOWED_METHODS]
    if unknown_methods:
        raise ValueError(
            f"This round only supports methods {ALLOWED_METHODS}; got unsupported "
            f"{unknown_methods}. LS/LS-QF/2SLS/RKHS are not implemented yet."
        )

    # Resolve every requested scenario up front (raises immediately for an
    # unknown/disabled one -- never buried as a per-repetition failure row).
    scenarios: Dict[str, ScenarioSpec] = {sid: get_scenario(sid) for sid in scenario_ids}

    output_dir = Path(output_dir)
    frozen_configs: Dict[str, Tuple[FrozenConfig, Dict[str, Any]]] = {}
    frozen_config_hashes: Dict[str, str] = {}
    for method in methods:
        if method not in METHODS_REQUIRING_FROZEN_CONFIG:
            continue  # oracle/parametric_proximal: no frozen config at all.
        for method_id in METHOD_FROZEN_IDS[method]:
            if method_id in frozen_configs:
                continue
            if method_id not in frozen_config_paths:
                raise KeyError(f"Missing frozen_config_paths entry for {method_id!r} (required by --methods {method}).")
            frozen, train_params = load_frozen_config(frozen_config_paths[method_id], method_id, n_splits)
            frozen_configs[method_id] = (frozen, train_params)
            frozen_config_hashes[method_id] = frozen.config_hash()

    run_config = build_run_config(
        run_stage, scenario_ids, methods, sample_sizes, repetitions, test_size, n_splits, base_seed,
        frozen_config_hashes,
    )
    ensure_run_config(output_dir, run_config)

    repetitions_csv = output_dir / "common" / "repetitions.csv"
    folds_csv = output_dir / "common" / "folds.csv"
    failures_csv = output_dir / "common" / "failures.csv"
    fold_diagnostics_csv = output_dir / "common" / "fold_diagnostics.csv"
    bridge_csv = output_dir / "e1" / "bridge_results.csv"
    effect_csv = output_dir / "e2" / "effect_results.csv"

    summary = {"attempted": 0, "succeeded": 0, "failed": 0}
    device = get_device()

    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        for n in sample_sizes:
            for repetition in range(1, repetitions + 1):
                rep_seeds = derive_repetition_seeds(base_seed, scenario_id, n, repetition)
                train_df = scenario.generate_train(n, rep_seeds.data_seed)
                test_df = scenario.generate_test(test_size, rep_seeds.test_seed)
                train_dataset = SGDDataset(csv_path='', df=train_df, device=device)
                test_dataset = SGDDataset(csv_path='', df=test_df, device=device)

                try:
                    fold_plan = build_fold_plan(train_dataset, n_splits, rep_seeds.split_seed)
                except ValueError as exc:
                    summary["attempted"] += len(methods)
                    summary["failed"] += len(methods)
                    _append_rows(failures_csv, [{
                        "run_stage": run_stage, "scenario": scenario_id, "n": n, "repetition": repetition,
                        "method": method, "variant": "", "failure_stage": "fold_plan",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    } for method in methods])
                    continue

                _append_rows(repetitions_csv, [{
                    "run_stage": run_stage, "scenario": scenario_id, "n": n, "repetition": repetition,
                    "data_seed": rep_seeds.data_seed, "test_seed": rep_seeds.test_seed,
                    "split_seed": rep_seeds.split_seed, "init_seed": rep_seeds.init_seed,
                    "data_hash": _dataframe_sha256(train_df), "test_hash": _dataframe_sha256(test_df),
                    "fold_plan_hash": fold_plan.fold_plan_hash,
                }])
                _append_rows(folds_csv, [
                    {"run_stage": run_stage, "scenario": scenario_id, "n": n, "repetition": repetition, **asdict(r)}
                    for r in fold_plan.records
                ])

                for method in methods:
                    summary["attempted"] += 1
                    h_train_params = q_train_params = None
                    if method in METHODS_REQUIRING_FROZEN_CONFIG:
                        h_id, q_id = METHOD_FROZEN_IDS[method]
                        _, h_train_params = frozen_configs[h_id]
                        _, q_train_params = frozen_configs[q_id]
                    key_row = {
                        "run_stage": run_stage, "scenario": scenario_id, "n": n,
                        "repetition": repetition, "method": method, "variant": "",
                    }
                    try:
                        bridge_row, effect_row, fold_diag_rows = compute_method_results(
                            method, train_dataset, test_dataset, fold_plan, rep_seeds,
                            h_train_params, q_train_params, scenario,
                        )
                    except Exception as exc:  # noqa: BLE001 -- deliberate: one method's failure must not stop the rest.
                        summary["failed"] += 1
                        _append_rows(failures_csv, [{
                            **key_row, "failure_stage": "fit_or_predict",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }])
                        continue
                    summary["succeeded"] += 1
                    _append_rows(bridge_csv, [{**key_row, **bridge_row}])
                    _append_rows(effect_csv, [{**key_row, **effect_row}])
                    _append_rows(fold_diagnostics_csv, [
                        {
                            "scenario_id": scenario_id, "repetition": repetition, "sample_size": n,
                            **row,
                        }
                        for row in fold_diag_rows
                    ])

    return summary
