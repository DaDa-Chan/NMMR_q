"""
HPO protocol orchestration: pilot data roles, the two-round Optuna
selection process, and the candidate-config artifact.

See plan/experiment_1/training_and_hpo_plan.md sections 5.1, 5.2, 6,
plan/experiment_1/hpo_selection_plan.md (the more detailed two-round
version of the same selection process) sections 3-10, and
plan/experiment_1/精简与config管理.md (candidate/frozen-config separation,
study-identity hardening, and record simplification).

File layout -- everything HPO ever writes automatically lives under one
`results_dir` per search space/run (精简与config管理.md section 3.1); HPO
code never writes to configs/sgd_frozen/ itself:
    data/pilot/{dgp_version}/scenario_{scenario}/seed_{base_seed}/
                                 select_n{n}.csv, confirm_n{n}.csv,
                                 manifest.json -- sample-size-keyed so
                                 e.g. select_n5000.csv and select_n10000.csv
                                 coexist under the same seed_{base_seed}
                                 directory without either overwriting the
                                 other (see ensure_pilot_dataset).
    <results_dir>/              search_config_snapshot.json (single method)
                                 or search_configs/{method}.json (multiple),
                                 optuna_studies/{method}_round1.db,
                                 {method}_round1_trials.csv,
                                 {method}_round2_finalists.csv,
                                 {method}_pilot_confirm_report.json,
                                 candidates/{method}_candidate_config.json
                                 (written regardless of confirm pass/fail;
                                 confirm_status/confirm_failure_reasons are
                                 recorded INSIDE the candidate itself).
    configs/sgd_frozen/          only ever written to by a human, who
                                 manually copies a passed candidate here
                                 once they've checked it (see
                                 plan section 4.3) -- no function in this
                                 module writes to this directory.

Callers (and tests) that want output kept out of the real data/pilot or
tune_results folders should pass an explicit directory of their own (e.g.
a tempdir) to every function below instead of relying on the defaults.
"""
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.ate.sgd_pv import generate_data
from src.data.ate.data_class import SGDDataset
from src.models.experiment_common import (
    _parse_optuna_space,
    nested_hpo_fold_split,
    insufficient_treatment_groups_reason,
    weighted_fold_mean,
    fold_score_dispersion,
    pick_simplest_within_one_se,
    select_top_k_candidates,
    build_fold_record,
    NoFiniteRound1TrialError,
    NoFiniteRound2CandidateError,
)
from src.utils import get_device, make_seed_bundle, set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fixed mapping, not a hash: numpy's SeedSequence entropy must be integers,
# so a method_id string is mapped to a small fixed int instead of being fed
# in directly.
METHOD_IDS = ("H-U", "H-V", "Q-U", "Q-V")
_METHOD_ID_TO_INT = {method_id: i for i, method_id in enumerate(METHOD_IDS)}
DEFAULT_ROUND2_INIT_SEEDS = (11, 22, 33)  # plan hpo_selection_plan.md section 7


def validate_method_ids(methods: List[str]) -> List[str]:
    """
    Validate a list of method IDs (e.g. from --methods) against METHOD_IDS,
    raising a clear error naming the bad entries instead of silently
    ignoring or mis-running them.
    """
    unknown = [m for m in methods if m not in METHOD_IDS]
    if unknown:
        raise ValueError(
            f"Unknown method id(s) {unknown}; expected a comma-separated "
            f"subset of {METHOD_IDS}."
        )
    return methods


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _method_slug(method_id: str) -> str:
    return method_id.lower().replace("-", "_")


# ---------------------------------------------------------------------------
# File layout: single source of truth for where each artifact lives
# ---------------------------------------------------------------------------

def pilot_data_dir() -> Path:
    """Real pilot_select/pilot_confirm data, alongside data/SGD, data/right_heart_catheterization."""
    return PROJECT_ROOT / "data" / "pilot"


def frozen_config_dir() -> Path:
    """
    Where a USER manually promotes a passed candidate to (精简与config管理.md
    section 3.2/4.3), alongside configs/sgd, configs/rhc. No function in
    this module ever writes here -- it exists purely as a path a human
    copies a candidate_config.json to by hand, after checking
    confirm_status=="passed" and everything else section 4.3 asks for.
    """
    return PROJECT_ROOT / "configs" / "sgd_frozen"


def tune_results_dir() -> Path:
    """Real HPO run records (trial tables, finalist tables, confirm reports)."""
    return PROJECT_ROOT / "tune_results"


def candidate_config_path(method_id: str, base_dir: Optional[Path] = None) -> Path:
    """
    The ONE place HPO ever writes this method's selected candidate --
    regardless of whether pilot-confirm passed or failed (精简与config管理.md
    section 4.1: "无论 confirm 通过还是失败，都只将 candidate 保存到当前
    results_dir/candidates/"). `confirm_status`/`confirm_failure_reasons`
    inside the saved FrozenConfig distinguish the two outcomes; promoting a
    passed one to configs/sgd_frozen/ is always a manual, human step.
    """
    base = Path(base_dir) if base_dir is not None else tune_results_dir()
    return base / "candidates" / f"{_method_slug(method_id)}_candidate_config.json"


def search_config_snapshot_path(results_dir: Path, method_id: Optional[str] = None) -> Path:
    """
    Human-readable copy of the exact search config this run used (section
    4.2). `method_id=None` -> results_dir/search_config_snapshot.json (a
    single-method run); otherwise -> results_dir/search_configs/{method}.json
    (a multi-method run, one snapshot file per method).
    """
    results_dir = Path(results_dir)
    if method_id is None:
        return results_dir / "search_config_snapshot.json"
    return results_dir / "search_configs" / f"{_method_slug(method_id)}.json"


def ensure_search_config_snapshot(config_bytes: bytes, snapshot_path: Path) -> None:
    """
    Write-once, byte-exact provenance snapshot of the search config a
    `results_dir` was first created against:
      - snapshot doesn't exist yet -> write `config_bytes` verbatim (no
        json.load/json.dump round-trip, which would silently reformat the
        file and defeat the point of a byte-exact snapshot);
      - snapshot exists and is byte-identical to `config_bytes` -> reuse
        it, no-op (this is the ordinary "resume this results_dir" case);
      - snapshot exists but differs -> raise immediately, WITHOUT touching
        the existing snapshot file and WITHOUT running any HPO for this
        method. A changed search config reusing an old `results_dir` is
        exactly the mistake this guards against -- the fix is a new
        `--results_dir`, not silently overwriting the provenance record of
        what the old trials in this study were actually searched over.
    """
    snapshot_path = Path(snapshot_path)
    if snapshot_path.exists():
        existing_bytes = snapshot_path.read_bytes()
        if existing_bytes == config_bytes:
            return
        raise ValueError(
            f"{snapshot_path} already records a DIFFERENT search config than the one "
            f"just loaded for this run; refusing to overwrite it or to run any HPO "
            f"for this method. If the search config genuinely changed, use a new "
            f"--results_dir instead of reusing this one."
        )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(config_bytes)


def optuna_study_db_path(method_id: str, base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else tune_results_dir()
    return base / "optuna_studies" / f"{_method_slug(method_id)}_round1.db"


def round1_trials_path(method_id: str, base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else tune_results_dir()
    return base / f"{_method_slug(method_id)}_round1_trials.csv"


def round2_finalists_path(method_id: str, base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else tune_results_dir()
    return base / f"{_method_slug(method_id)}_round2_finalists.csv"


def pilot_confirm_report_path(method_id: str, base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else tune_results_dir()
    return base / f"{_method_slug(method_id)}_pilot_confirm_report.json"


# ---------------------------------------------------------------------------
# Plan 5.1 / hpo_selection_plan.md section 3: pilot data roles
# ---------------------------------------------------------------------------

@dataclass
class PilotDatasets:
    """
    pilot_select and pilot_confirm are independently generated,
    independently seeded, and (once persisted via `save_dir`) never
    regenerated on the fly -- any later call for the SAME (scenario,
    base_seed, n) reuses the existing file instead (see ensure_pilot_dataset).
    """
    select_dataset: Any
    confirm_dataset: Any
    select_data_seed: int
    confirm_data_seed: int
    select_sha256: str
    confirm_sha256: str
    n_pilot_select: int
    n_pilot_confirm: int
    scenario: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_sha256(df: pd.DataFrame) -> str:
    """Same hash a saved CSV would get (df.to_csv(path, index=False) writes
    the exact same bytes as df.to_csv(index=False).encode()) -- used so an
    ephemeral (root=None) dataset still gets a real, comparable SHA-256
    instead of a file-backed one."""
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


DEFAULT_DGP_VERSION = "sgd_v1"
_LEGACY_PILOT_FILENAMES = ("pilot_select.csv", "pilot_confirm.csv", "pilot_seeds.json")


def pilot_seed_dir(
    dgp_version: str, scenario: int, base_seed: int, root: Optional[Path] = None,
) -> Path:
    """data/pilot/{dgp_version}/scenario_{scenario}/seed_{base_seed}/ -- the
    one directory that holds every select_n*/confirm_n* file + manifest.json
    for a given (dgp_version, scenario, base_seed) triple."""
    base = Path(root) if root is not None else pilot_data_dir()
    return base / dgp_version / f"scenario_{scenario}" / f"seed_{base_seed}"


def _pilot_dataset_key(role: str, n: int) -> str:
    return f"{role}_n{n}"


def _pilot_manifest_path(seed_dir: Path) -> Path:
    return seed_dir / "manifest.json"


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def _save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def cleanup_or_reject_legacy_pilot_files(root: Path) -> List[Path]:
    """
    The old flat pilot_select.csv/pilot_confirm.csv/pilot_seeds.json format
    is replaced by the sample-size-keyed structure above. If `root` is
    exactly this project's own default pilot_data_dir() and any of those
    three exact files are sitting directly under it, delete just those
    three files (never the directory, never anything else) and return what
    was removed. If `root` is some OTHER caller-specified directory (a
    custom --data_dir, or a test's own tempdir) and legacy files happen to
    be there, refuse instead of silently deleting outside the project's own
    default location -- the caller must migrate or clean those up by hand.
    """
    root = Path(root)
    found = [root / name for name in _LEGACY_PILOT_FILENAMES if (root / name).exists()]
    if not found:
        return []
    if root.resolve() != pilot_data_dir().resolve():
        raise ValueError(
            f"Legacy flat-format pilot files {found} found under {root}, which is "
            f"not this project's default pilot_data_dir() ({pilot_data_dir()}). "
            f"Refusing to auto-delete files outside the default location -- "
            f"please migrate or clean these up manually."
        )
    for path in found:
        path.unlink()
    return found


def ensure_pilot_dataset(
    role: str,
    n: int,
    scenario: int,
    base_seed: int,
    dgp_version: str = DEFAULT_DGP_VERSION,
    root: Optional[Path] = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Get the pilot dataset for (role, n, scenario, base_seed), reusing an
    existing file+manifest record when they already agree, generating one
    when neither exists, and refusing (rather than guessing) when the file
    and the manifest disagree about each other's existence or content.

    `role` is "select" or "confirm"; each role gets its own data_seed,
    derived from (base_seed, role) only -- independent of `n`, so
    select_n5000 and select_n10000 under the same seed_{base_seed} draw
    from the SAME underlying stream (just truncated to a different length),
    and select/confirm never share a seed.

    `root=None` (the default) is the ephemeral mode: nothing touches disk,
    a fresh dataset is generated every call (used by tests that want no
    file-system footprint at all). `root` given persists to
    `pilot_seed_dir(dgp_version, scenario, base_seed, root)/{role}_n{n}.csv`
    and that seed directory's shared manifest.json, applying these rules:
      1. file + manifest record both exist and agree (role, n, data_seed,
         row count, SHA-256) -> reuse as-is.
      2. neither exists -> generate and record in the manifest.
      3. file exists but manifest has no record for this key -> error.
      4. manifest has a record but the file is missing -> error.
      5. file and manifest both exist but disagree -> error (never guess
         which one is "right").
      6. generating a different `n` never touches another `n`'s file or
         manifest entry (each lives under its own key).

    `force_regenerate=True` unconditionally regenerates (and overwrites)
    ONLY this exact (role, n) pair -- printing the old/new SHA-256 when an
    existing file is being replaced -- and never touches any other (role, n).

    Returns {"df", "data_seed", "sha256", "path"} ("path" is None in the
    ephemeral root=None mode).
    """
    if role not in ("select", "confirm"):
        raise ValueError(f"role must be 'select' or 'confirm', got {role!r}")

    role_idx = 0 if role == "select" else 1
    data_seed = make_seed_bundle(base_seed, role_idx).data_seed

    if root is None:
        df = generate_data(n_samples=n, scenario=scenario, seed=data_seed)
        return {"df": df, "data_seed": data_seed, "sha256": _dataframe_sha256(df), "path": None}

    seed_dir = pilot_seed_dir(dgp_version, scenario, base_seed, root)
    manifest_path = _pilot_manifest_path(seed_dir)
    manifest = _load_manifest(manifest_path)
    key = _pilot_dataset_key(role, n)
    filename = f"{key}.csv"
    file_path = seed_dir / filename
    record = manifest.get("datasets", {}).get(key)
    file_exists = file_path.exists()

    def _write(df: pd.DataFrame) -> str:
        seed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        file_hash = _sha256_file(file_path)
        manifest["dgp_version"] = dgp_version
        manifest["scenario"] = scenario
        manifest["base_seed"] = base_seed
        manifest.setdefault("datasets", {})
        manifest["datasets"][key] = {
            "role": role, "n": n, "data_seed": data_seed, "path": filename, "sha256": file_hash,
        }
        _save_manifest(manifest_path, manifest)
        return file_hash

    if force_regenerate:
        old_hash = None
        if file_exists:
            old_hash = _sha256_file(file_path)
            print(f"Regenerating {filename}")
            print(f"Old SHA-256: {old_hash}")
        df = generate_data(n_samples=n, scenario=scenario, seed=data_seed)
        new_hash = _write(df)
        if file_exists:
            print(f"New SHA-256: {new_hash}")
        return {"df": df, "data_seed": data_seed, "sha256": new_hash, "path": file_path}

    if file_exists and record is not None:
        actual_hash = _sha256_file(file_path)
        df = pd.read_csv(file_path)
        errors = []
        if record.get("role") != role:
            errors.append(f"role mismatch: manifest={record.get('role')!r} vs expected={role!r}")
        if record.get("n") != n:
            errors.append(f"n mismatch: manifest={record.get('n')} vs expected={n}")
        if len(df) != n:
            errors.append(f"row count mismatch: file has {len(df)} rows, expected {n}")
        if record.get("data_seed") != data_seed:
            errors.append(f"data_seed mismatch: manifest={record.get('data_seed')} vs expected={data_seed}")
        if record.get("sha256") != actual_hash:
            errors.append(f"sha256 mismatch: manifest={record.get('sha256')} vs actual={actual_hash}")
        if errors:
            raise ValueError(f"{file_path} inconsistent with {manifest_path}: " + "; ".join(errors))
        return {"df": df, "data_seed": record["data_seed"], "sha256": actual_hash, "path": file_path}

    if file_exists and record is None:
        raise ValueError(
            f"{file_path} exists but {manifest_path} has no record for {key!r}; "
            f"refusing to guess whether it is reusable -- fix the manifest or "
            f"remove the stray file."
        )

    if not file_exists and record is not None:
        raise ValueError(
            f"{manifest_path} records {key!r} but {file_path} is missing; "
            f"refusing to silently regenerate it (the regenerated content "
            f"might not match what was recorded)."
        )

    # Neither exists: generate fresh and record it, without touching any
    # other (role, n) key already in this seed_dir's manifest.
    df = generate_data(n_samples=n, scenario=scenario, seed=data_seed)
    file_hash = _write(df)
    return {"df": df, "data_seed": data_seed, "sha256": file_hash, "path": file_path}


def generate_pilot_datasets(
    base_seed: int,
    n_pilot_select: int = 5000,
    n_pilot_confirm: int = 5000,
    scenario: int = 1,
    save_dir: Optional[Path] = None,
    force_regenerate: bool = False,
    dgp_version: str = DEFAULT_DGP_VERSION,
) -> PilotDatasets:
    """
    Get the (select, confirm) pilot dataset pair for this (scenario,
    base_seed), each looked up / generated / validated independently by
    `ensure_pilot_dataset` under its own sample-size-keyed filename, so
    e.g. requesting n_pilot_select=10000 after a prior n_pilot_select=5000
    run adds select_n10000.csv alongside select_n5000.csv rather than
    replacing it.

    `save_dir=None` (the default) is fully ephemeral -- nothing is read
    from or written to disk; callers that want the real data/pilot/ folder
    must pass `save_dir=pilot_data_dir()` explicitly, and tests should pass
    their own tempdir to stay isolated from real pilot data.

    If `save_dir` is given and is exactly this project's own default
    pilot_data_dir() with legacy pilot_select.csv/pilot_confirm.csv/
    pilot_seeds.json sitting directly in it, those three files are removed
    first (see cleanup_or_reject_legacy_pilot_files); if `save_dir` is some
    other directory with those legacy files in it, this raises instead.
    """
    if save_dir is not None:
        deleted = cleanup_or_reject_legacy_pilot_files(Path(save_dir))
        if deleted:
            print(f"[pilot data] Removed legacy flat-format files: {[str(p) for p in deleted]}")

    device = get_device()
    select = ensure_pilot_dataset(
        "select", n_pilot_select, scenario, base_seed, dgp_version, save_dir, force_regenerate,
    )
    confirm = ensure_pilot_dataset(
        "confirm", n_pilot_confirm, scenario, base_seed, dgp_version, save_dir, force_regenerate,
    )

    return PilotDatasets(
        select_dataset=SGDDataset(csv_path='', df=select["df"], device=device),
        confirm_dataset=SGDDataset(csv_path='', df=confirm["df"], device=device),
        select_data_seed=select["data_seed"],
        confirm_data_seed=confirm["data_seed"],
        select_sha256=select["sha256"],
        confirm_sha256=confirm["sha256"],
        n_pilot_select=n_pilot_select,
        n_pilot_confirm=n_pilot_confirm,
        scenario=scenario,
    )


# ---------------------------------------------------------------------------
# Frozen config artifact (plan 6.2 / hpo_selection_plan.md section 10)
# ---------------------------------------------------------------------------

@dataclass
class FrozenConfig:
    """
    The immutable artifact for one method (H-U/H-V/Q-U/Q-V) once the
    two-round selection (hpo_selection_plan.md sections 6-8) is finished.
    This single file is both the archival record (every field the plan
    lists) AND sufficient on its own to resume training -- see
    `to_train_params`.
    """
    method_id: str = ""
    network_depth: Optional[int] = None
    network_width: Optional[int] = None
    learning_rate: Optional[float] = None
    l2_penalty: Optional[float] = None
    batch_size: Optional[int] = None
    training_objective: str = ""  # "U" or "V"
    hpo_scoring_objective: str = "held-out V-statistic moment risk (plan 5.4 / hpo_selection_plan 6.3-6.4)"
    kernel_rule: str = "median heuristic, frozen per fold from gradient_train only (plan 3.4/5.3)"
    kernel_bandwidth_multiplier: Optional[float] = None
    # Q only (None for H): the same hard threshold Round 1/2 already
    # screened every candidate's predictions against (pilot HPO fix
    # request item 3). pilot-confirm's decide_pilot_confirm() checks the
    # final OOF Q predictions against this SAME value -- not a separately
    # tunable "confirm threshold" (精简与config管理.md section 6.5 item 5).
    q_hard_threshold: Optional[float] = None
    n_splits: Optional[int] = None
    max_epochs: Optional[int] = None
    patience: Optional[int] = None
    early_stopping_min_delta: Optional[float] = None
    scheduler_factor: Optional[float] = None
    scheduler_patience: Optional[int] = None
    scheduler_min_lr: Optional[float] = None
    preprocessing_rule: str = "none (this pipeline has no feature scaler)"
    pilot_select_data_seed: Optional[int] = None
    split_seed: Optional[int] = None
    tuning_seed: Optional[int] = None
    optuna_sampler: str = "TPESampler"
    # 精简与config管理.md Phase A item 4: these two are distinct and neither
    # should be guessed from the other -- "additional" is just what THIS
    # invocation asked _run_optuna_tuning to run; "total" is how many trials
    # the persisted study actually holds afterward (0 extra on top of
    # whatever was already there when storage_dir is given, or simply
    # equal to "additional" for an ephemeral in-memory study).
    round1_additional_trials: Optional[int] = None
    round1_total_trials: Optional[int] = None
    round1_top_k: Optional[int] = None
    round2_init_seeds: List[int] = field(default_factory=list)
    software_version: str = field(default_factory=_get_git_commit)
    selection_rule: str = "one-standard-error over Round 2's multi-seed scores (hpo_selection_plan.md section 8)"
    selected_round1_trial_number: Optional[int] = None
    pure_min_round1_trial_number: Optional[int] = None
    # "provisional" until pilot-confirm has run; "confirmed" only once it has
    # passed (pilot HPO fix request item 5/6) -- "failed" if it ran and did
    # not pass. Every candidate, regardless of this status, is written to
    # candidate_config_path() (精简与config管理.md section 4.1) -- promoting
    # a "confirmed" one to configs/sgd_frozen/ is a manual, human step, not
    # anything this module does automatically.
    status: str = "provisional"
    # confirm_status/confirm_failure_reasons mirror decide_pilot_confirm()'s
    # own "passed"/"failed" + reasons (精简与config管理.md section 4.1: "candidate
    # 中记录 confirm_status=passed|failed 和失败原因"), kept alongside `status`
    # (this artifact's own provisional/confirmed/failed lifecycle label) so
    # the saved candidate file is self-describing without cross-referencing
    # the separate pilot_confirm_report.json.
    confirm_status: Optional[str] = None
    confirm_failure_reasons: List[str] = field(default_factory=list)

    def config_hash(self) -> str:
        """Checksum of the frozen artifact (plan 6.2: "最终 frozen config 的校验哈希")."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_train_params(self, model_kind: str) -> Dict[str, Any]:
        """
        Reconstruct the literal train_params dict the trainer needs from
        this one frozen artifact (no separate "resolved params" file is
        generated -- this is the single necessary file per plan 6.2's
        "每个方法保存一个不可变配置文件").
        """
        return {
            "model_name": "nmmr_q" if model_kind == "Q" else "nmmr_h",
            "loss_name": f"{self.training_objective}_statistic",
            "use_u_statistic": self.training_objective == "U",
            "n_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_min_lr": self.scheduler_min_lr,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "n_splits": self.n_splits,
            "network_depth": self.network_depth,
            "network_width": self.network_width,
            "learning_rate": self.learning_rate,
            "l2_penalty": self.l2_penalty,
            "kernel_bandwidth_multiplier": self.kernel_bandwidth_multiplier,
            "log_metrics": False,
        }

    def save(self, path: Path) -> None:
        payload = asdict(self)
        payload["config_hash"] = self.config_hash()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


def validate_frozen_config(train_params: Dict[str, Any]) -> None:
    """
    Loading a frozen config for the FORMAL experiment must fail loudly if it
    still looks like an HPO config -- a leftover Optuna search space dict,
    or an `hpo_n_trials` key -- rather than single resolved values. (The
    `n_epochs += 200` hack the plan also warns about was removed entirely
    from this codebase in Phase A; there is no longer a code path that
    could silently reintroduce it.)
    """
    _, search_space = _parse_optuna_space(train_params)
    if search_space:
        raise ValueError(
            f"Frozen config still has an Optuna search space for {list(search_space)}; "
            "a frozen config must carry single resolved values, not search ranges."
        )
    if "hpo_n_trials" in train_params:
        raise ValueError("Frozen config must not contain 'hpo_n_trials' (an HPO-only key).")


def decide_pilot_confirm(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicit pilot-confirm pass/fail judgment (精简与config管理.md section
    6.5): a FIXED checklist of hard failure conditions, none of them
    hardware/runtime-budget related --
      1. every fold completed with a valid best checkpoint;
      2. the ATE estimate (and every per-fold ATE) is finite;
      3. held-out moment risk is finite;
      4. the H/Q predictions contain no NaN/Inf;
      5. (Q only) no prediction exceeds this method's pre-specified
         `q_hard_threshold` (the SAME one Round 1/2 already screened
         candidates against -- see FrozenConfig.q_hard_threshold).
    Per-fold/total training time, how often training hit max_epochs, and
    Q's extreme-value proportions remain in the report as diagnostics the
    reader should look at, but none of them gates pass/fail any more (plan:
    "不把硬件速度、每 epoch 耗时等运维信息作为统计方法的通过门槛"). Returns
    {"status": "passed"|"failed", "failure_reasons": [...]}. Only a
    "passed" status may ever produce a confirmed candidate.
    """
    reasons: List[str] = []

    if "exception" in report:
        reasons.append(f"pilot_confirm_exception: {report['exception']}")
        return {"status": "failed", "failure_reasons": reasons}

    if not report.get("all_folds_restored_best_checkpoint", False):
        reasons.append("not_all_folds_restored_best_checkpoint")
    if not report.get("ate_all_finite", False):
        reasons.append("ate_not_finite")
    if not report.get("moment_risk_all_finite", False):
        reasons.append("moment_risk_not_all_finite")
    if not report.get("predictions_all_finite", False):
        reasons.append("predictions_not_finite")

    q_hard_threshold = report.get("q_hard_threshold")
    q_max_abs = report.get("q_max_abs")
    if q_hard_threshold is not None and q_max_abs is not None and q_max_abs > q_hard_threshold:
        reasons.append(f"q_exceeds_hard_threshold(max_q={q_max_abs:.2f})")

    status = "passed" if not reasons else "failed"
    return {"status": status, "failure_reasons": reasons}


def build_trial_table(study, method_id: Optional[str] = None) -> pd.DataFrame:
    """
    One row per (trial, fold) -- pilot HPO fix request item 7: every field
    a fold's outcome needs for audit (hyperparameters, seeds, score, best
    epoch, stop reason, failure reason, etc.), not just a per-trial
    summary. A trial that failed before producing any `fold_records` (e.g.
    an unexpected error outside the per-fold try/except) still gets one row
    carrying its own `failure_reason` and no fold-level fields.
    """
    rows = []
    for trial in study.trials:
        base_row = {
            "method_id": method_id,
            "trial_number": trial.number,
            "state": str(trial.state),
            "trial_value": trial.value,
        }
        fold_records = trial.user_attrs.get("fold_records")
        if fold_records:
            for rec in fold_records:
                rows.append({**base_row, **rec})
        else:
            rows.append({
                **base_row,
                **{f"param.{k}": v for k, v in trial.params.items()},
                "failure_reason": trial.user_attrs.get("failure_reason"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Round 2: multi-seed re-evaluation of Round 1's finalists
# (hpo_selection_plan.md section 7)
# ---------------------------------------------------------------------------

def reevaluate_candidate_with_seeds(
    method_id: str,
    params: Dict[str, Any],
    dataset,
    data_configs: Dict[str, Any],
    n_splits: int,
    split_seed: int,
    init_seeds: Tuple[int, ...] = DEFAULT_ROUND2_INIT_SEEDS,
) -> Dict[str, Any]:
    """
    Re-run ONE candidate's hyperparameters across `init_seeds` x `n_splits`
    folds -- the SAME fold assignment/nested split as Round 1 (same
    `split_seed`), only the network initialization differs -- to check
    robustness to random initialization before final selection.

    Pilot HPO fix request item 3: ANY seed/fold hitting a pre-specified
    failure condition (insufficient treatment groups, no valid checkpoint,
    NaN/Inf during training or in the score, a training exception, or --
    for Q -- predictions over a fixed hard threshold) makes this
    CANDIDATE's overall `value` +inf, same as Round 1. The previous
    behaviour of silently dropping only the failed (seed, fold) combos and
    averaging whatever succeeded is exactly what item 3 forbids.

    Returns a dict with `value` (the weighted mean score, same key name
    `pick_simplest_within_one_se` expects; +inf if ANY combo failed),
    `fold_se`, `fold_sd`, `n_scores`, `total_train_seconds`,
    `failure_reasons`, `fold_records` (item 7's per-(seed,fold) record).
    """
    model_kind, _ = method_id.split("-")
    if model_kind == "Q":
        from src.models.NMMRq.nmmr_q_experiments import (
            _create_trainer, _compute_unified_q_score as compute_score,
            _q_predictions_by_treatment, _max_abs_q, _q_distribution_stats,
            DEFAULT_Q_HARD_THRESHOLD, DEFAULT_MAX_FOLD_TRAIN_SECONDS,
        )
    else:
        from src.models.NMMRh.nmmr_h_experiments import (
            _create_trainer, _compute_unified_h_score as compute_score,
            DEFAULT_MAX_FOLD_TRAIN_SECONDS,
        )
    from torch.utils.data import DataLoader, Subset
    from src.utils import empty_device_cache, get_device

    batch_size = params["batch_size"]
    q_hard_threshold = float(params.get("q_hard_threshold", DEFAULT_Q_HARD_THRESHOLD)) if model_kind == "Q" else None
    max_fold_train_seconds = float(params.get("max_fold_train_seconds", DEFAULT_MAX_FOLD_TRAIN_SECONDS))
    folds = nested_hpo_fold_split(dataset, n_splits, split_seed)

    scores: List[float] = []
    sizes: List[int] = []
    train_seconds: List[float] = []
    failure_reasons: List[str] = []
    fold_records: List[Dict[str, Any]] = []
    any_failure = False

    def record_failure(fold_idx, init_seed, reason, train_result=None, extra=None):
        failure_reasons.append(f"seed{init_seed}_fold{fold_idx}_{reason}")
        fold_records.append(build_fold_record(
            fold_idx, init_seed, None, params, float("inf"), train_result,
            failure_reason=reason, extra={"round2_seed": init_seed, **(extra or {})},
        ))

    for init_seed in init_seeds:
        for fold_idx, (gradient_train_idx, earlystop_val_idx, score_idx) in enumerate(folds, start=1):
            reason = insufficient_treatment_groups_reason(
                dataset,
                {
                    "gradient_train": gradient_train_idx,
                    "earlystop_val": earlystop_val_idx,
                    "score": score_idx,
                },
            )
            if reason is not None:
                record_failure(fold_idx, init_seed, reason)
                any_failure = True
                continue

            fold_init_seed = make_seed_bundle(init_seed, fold_idx).init_seed
            set_random_seed(fold_init_seed)
            trainer = _create_trainer(data_configs, params, split_seed, None, "sgd")

            train_loader = DataLoader(
                Subset(dataset, gradient_train_idx.tolist()), batch_size=batch_size, shuffle=True, num_workers=0
            )
            earlystop_loader = DataLoader(
                Subset(dataset, earlystop_val_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=0
            )

            try:
                model, train_result = trainer.train(train_loader, earlystop_loader)

                if train_result.stop_reason == "no_valid_checkpoint":
                    record_failure(fold_idx, init_seed, "no_valid_checkpoint", train_result)
                    any_failure = True
                    continue
                if train_result.n_nonfinite_train > 0 or train_result.n_nonfinite_val > 0:
                    record_failure(fold_idx, init_seed, "nonfinite_during_training", train_result)
                    any_failure = True
                    continue

                q_stats: Dict[str, Any] = {}
                if model_kind == "Q":
                    q_by_treatment = _q_predictions_by_treatment(trainer, model, dataset, score_idx.tolist())
                    max_q = _max_abs_q(q_by_treatment)
                    q_stats = _q_distribution_stats(q_by_treatment)
                    if max_q > q_hard_threshold:
                        record_failure(fold_idx, init_seed, f"q_exceeds_hard_threshold(max_q={max_q:.2f})",
                                       train_result, extra=q_stats)
                        any_failure = True
                        continue
                if train_result.train_seconds > max_fold_train_seconds:
                    record_failure(fold_idx, init_seed, f"timeout(train_seconds={train_result.train_seconds:.1f})",
                                   train_result, extra=q_stats)
                    any_failure = True
                    continue

                score, n = compute_score(trainer, model, dataset, score_idx.tolist())
                if not math.isfinite(score):
                    record_failure(fold_idx, init_seed, "nonfinite_score", train_result, extra=q_stats)
                    any_failure = True
                    continue

                scores.append(score)
                sizes.append(n)
                train_seconds.append(train_result.train_seconds)
                fold_records.append(build_fold_record(
                    fold_idx, fold_init_seed, None, params, score, train_result,
                    failure_reason=None, extra={"round2_seed": init_seed, **q_stats},
                ))
            except Exception as e:
                record_failure(fold_idx, init_seed, f"exception: {e}")
                any_failure = True
            finally:
                del train_loader, earlystop_loader
                empty_device_cache(get_device())

    total_train_seconds = sum(train_seconds)
    if any_failure or not scores:
        return {
            "value": float("inf"), "fold_sd": 0.0, "fold_se": 0.0,
            "n_scores": len(scores), "total_train_seconds": total_train_seconds,
            "failure_reasons": failure_reasons, "fold_records": fold_records,
        }

    mean_score = weighted_fold_mean(scores, sizes)
    sd, se = fold_score_dispersion(scores)
    return {
        "value": mean_score, "fold_sd": sd, "fold_se": se,
        "n_scores": len(scores), "total_train_seconds": total_train_seconds,
        "failure_reasons": failure_reasons, "fold_records": fold_records,
    }


def _search_space_sha256(search_space: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(search_space, sort_keys=True, default=str).encode()).hexdigest()


# Fixed description of Round 1's held-out scoring rule (plan 5.4:
# _compute_unified_q_score/_compute_unified_h_score, always V-statistic
# moment risk regardless of this method's own training objective). Bump
# this string if that scoring rule itself ever changes, so an old study
# computed under the previous rule is never silently extended under a new
# one (精简与config管理.md section 6.1).
ROUND1_SCORING_RULE = "unified_v_statistic_moment_risk_v1"


def _prepare_persistent_study(
    storage: str,
    study_name: str,
    method_id: str,
    pilot_select_sha256: Optional[str],
    search_space: Dict[str, Any],
    n_splits: Optional[int] = None,
    n_epochs: Optional[Any] = None,
    patience: Optional[Any] = None,
    early_stopping_min_delta: Optional[Any] = None,
    split_seed: Optional[int] = None,
) -> int:
    """
    Pilot HPO fix (round 2) item 6 / 精简与config管理.md Phase A item 3: the
    FIRST time a study is created at (storage, study_name), record its full
    identity -- method_id, pilot-select file SHA-256, search-space
    SHA-256, plus every training/scoring setting that changes whether
    trials are even comparable (n_splits, n_epochs, patience,
    early_stopping_min_delta, split_seed, and the Round 1 scoring rule) --
    as study user attrs. Every later call that resumes the same (storage,
    study_name) must match ALL of these, or this raises -- refusing to let
    a different method/pilot-data/search-space/training-setting (e.g. a
    15-epoch smoke run vs. a 300-epoch formal run) accidentally append
    trials into the same study. On mismatch, nothing is deleted or
    overwritten; the caller must use a new `--results_dir`.

    Also returns the trial count already stored (0 for a brand new study),
    for the "Existing trials / Additional trials requested / Expected
    total" log (item 5).
    """
    import optuna

    identity = {
        "method_id": method_id,
        "pilot_select_sha256": pilot_select_sha256,
        "search_space_sha256": _search_space_sha256(search_space),
        "n_splits": n_splits,
        "n_epochs": n_epochs,
        "patience": patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "split_seed": split_seed,
        "round1_scoring_rule": ROUND1_SCORING_RULE,
    }
    study = optuna.create_study(storage=storage, study_name=study_name, load_if_exists=True)
    # user_attrs (not system_attrs, deprecated since Optuna 3.1 and removed
    # in 5.0) persisted on the Study itself -- distinct from each trial's
    # own user_attrs (failure_reason, fold_records, etc.).
    existing_identity = study.user_attrs
    if "method_id" not in existing_identity:
        for k, v in identity.items():
            study.set_user_attr(k, v if v is not None else "")
    else:
        mismatches = {
            k: (existing_identity.get(k), identity[k])
            for k in identity
            if existing_identity.get(k) != (identity[k] if identity[k] is not None else "")
        }
        if mismatches:
            raise ValueError(
                f"Study {study_name!r} at {storage} was created for a different "
                f"experiment (mismatched {list(mismatches)}: {mismatches}); refusing "
                f"to mix trials from a different method/pilot-select dataset/search "
                f"space into the same study. Use a new --results_dir for this run "
                f"instead -- nothing has been deleted or overwritten."
            )
    return len(study.trials)


# ---------------------------------------------------------------------------
# Full two-round selection for one method, and all four methods together
# (plan 5.2 / hpo_selection_plan.md sections 4, 6-8)
# ---------------------------------------------------------------------------

def run_hpo_study_for_method(
    method_id: str,
    pilot_select_dataset,
    pilot_select_data_seed: int,
    data_configs: Dict[str, Any],
    base_train_params: Dict[str, Any],
    search_space: Dict[str, Any],
    n_splits: int,
    n_trials: int,
    base_seed: int,
    top_k: int = 5,
    round2_init_seeds: Tuple[int, ...] = DEFAULT_ROUND2_INIT_SEEDS,
    storage_dir: Optional[Path] = None,
    pilot_select_sha256: Optional[str] = None,
) -> Tuple[FrozenConfig, pd.DataFrame, pd.DataFrame]:
    """
    Full two-round selection for ONE method (H-U, H-V, Q-U, or Q-V -- never
    mixed with another method's trials, plan 5.2), on the shared
    `pilot_select_dataset` so all four methods are compared on identical
    data:

    Round 1 (hpo_selection_plan.md section 6): one independent Optuna study,
      `n_trials` candidates, each scored by the unified held-out moment risk
      averaged over `n_splits` folds (single initialization per fold).
      Ranked by raw mean score; keep the best `top_k`.
    Round 2 (section 7): re-evaluate each of those `top_k` candidates across
      `round2_init_seeds` x `n_splits` folds (same fold IDs as Round 1, only
      initialization differs), to check robustness to random init.
    Final selection (section 8): one-SE rule applied to Round 2's richer
      per-candidate score distribution, not Round 1's single-seed score.

    Returns (frozen_config, round1_trial_table, round2_finalists_table) --
    `frozen_config`'s `status` is "provisional" here, before pilot-confirm
    has run. The caller persists these (see save_pipeline_artifacts, or
    run_and_confirm_method below for the full pipeline including confirm).
    """
    if method_id not in _METHOD_ID_TO_INT:
        raise ValueError(f"Unknown method_id {method_id!r}; expected one of {METHOD_IDS}")
    model_kind, stat_kind = method_id.split("-")
    expected_u_statistic = (stat_kind == "U")
    if bool(base_train_params.get("use_u_statistic", False)) != expected_u_statistic:
        raise ValueError(
            f"{method_id} implies use_u_statistic={expected_u_statistic}, but "
            f"base_train_params has use_u_statistic={base_train_params.get('use_u_statistic')}."
        )

    from src.models.NMMRq.nmmr_q_experiments import _run_optuna_tuning as run_q_tuning
    from src.models.NMMRh.nmmr_h_experiments import _run_optuna_tuning as run_h_tuning
    runner = run_q_tuning if model_kind == "Q" else run_h_tuning

    seeds = make_seed_bundle(base_seed, _METHOD_ID_TO_INT[method_id])

    # hpo_n_trials is HPO-management metadata (how many trials to run), not a
    # model hyperparameter; `n_trials` above is the actual trial count this
    # call uses. Strip it so it can never leak into a candidate's params
    # (validate_frozen_config/to_train_params expect single resolved values).
    base_train_params = {k: v for k, v in base_train_params.items() if k != "hpo_n_trials"}

    # Pilot HPO fix request item 7 (round 2 items 5/6): persist the Optuna
    # study to SQLite when a storage_dir is given, so an interrupted run can
    # resume (load_if_exists) and ADD trials instead of starting over, and
    # guard against a different method/pilot-data/search-space silently
    # appending trials into the same study.
    storage = study_name = None
    existing_trial_count = 0
    if storage_dir is not None:
        db_path = optuna_study_db_path(method_id, storage_dir)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"
        study_name = f"{_method_slug(method_id)}_round1"
        existing_trial_count = _prepare_persistent_study(
            storage=storage, study_name=study_name, method_id=method_id,
            pilot_select_sha256=pilot_select_sha256, search_space=search_space,
            n_splits=n_splits, n_epochs=base_train_params.get("n_epochs"),
            patience=base_train_params.get("patience"),
            early_stopping_min_delta=base_train_params.get("early_stopping_min_delta"),
            split_seed=seeds.split_seed,
        )
    print(f"[{method_id}] Existing trials: {existing_trial_count}")
    print(f"[{method_id}] Additional trials requested: {n_trials}")
    print(f"[{method_id}] Expected total after this run: {existing_trial_count + n_trials}")

    # --- Round 1: Optuna TPE screening ---
    # 精简与config管理.md Phase A item 1/2: the "every trial is non-finite"
    # check itself now lives INSIDE _run_optuna_tuning, raising
    # NoFiniteRound1TrialError BEFORE it ever prints a misleading "best
    # params" line for a value=inf trial. Nothing left to check here except
    # the degenerate "no search space at all" case (study is None).
    _r1_best_params, study, _r1_selected, _r1_best = runner(
        data_name="sgd",
        dataset=pilot_select_dataset,
        n_splits=n_splits,
        n_trials=n_trials,
        base_train_params=base_train_params,
        search_space=search_space,
        data_configs=data_configs,
        random_seed=seeds.split_seed,
        tuning_seed=seeds.tuning_seed,
        return_study=True,
        storage=storage,
        study_name=study_name,
        existing_trial_count=existing_trial_count,
    )
    if study is None:
        raise NoFiniteRound1TrialError(
            f"{method_id}: Round 1 did not run (empty search space); nothing to select."
        )
    round1_table = build_trial_table(study, method_id=method_id)
    top_k_trials = select_top_k_candidates(study, k=top_k)

    # --- Round 2: multi-seed re-evaluation of the top_k finalists ---
    round2_candidates = []
    round2_fold_rows: List[Dict[str, Any]] = []
    for trial in top_k_trials:
        candidate_params = {**base_train_params, **trial.params}
        agg = reevaluate_candidate_with_seeds(
            method_id=method_id, params=candidate_params, dataset=pilot_select_dataset,
            data_configs=data_configs, n_splits=n_splits, split_seed=seeds.split_seed,
            init_seeds=round2_init_seeds,
        )
        for rec in agg.get("fold_records", []):
            round2_fold_rows.append({
                "method_id": method_id, "round1_trial_number": trial.number, **rec,
            })
        round2_candidates.append({
            "round1_trial_number": trial.number,
            "network_depth": candidate_params.get("network_depth"),
            "network_width": candidate_params.get("network_width"),
            "l2_penalty": candidate_params.get("l2_penalty"),
            "learning_rate": candidate_params.get("learning_rate"),
            "batch_size": candidate_params.get("batch_size"),
            "kernel_bandwidth_multiplier": candidate_params.get("kernel_bandwidth_multiplier"),
            **agg,
            "_params": candidate_params,
        })

    # One row per (candidate, seed, fold) -- pilot HPO fix request item 7 --
    # not just one aggregate row per candidate.
    round2_table = pd.DataFrame(round2_fold_rows)

    selected, pure_min = pick_simplest_within_one_se(round2_candidates)
    if not math.isfinite(selected["value"]):
        raise NoFiniteRound2CandidateError(
            f"{method_id}: every Round 2 top_k candidate failed (value=inf) -- "
            "no usable candidate survived multi-seed re-evaluation. Check the "
            "data/config (pilot sample too small? thresholds too strict?) and "
            "re-run; do not silently proceed with a candidate that never passed."
        )
    final_params = selected["_params"]

    q_hard_threshold = None
    if model_kind == "Q":
        from src.models.NMMRq.nmmr_q_experiments import DEFAULT_Q_HARD_THRESHOLD
        q_hard_threshold = final_params.get("q_hard_threshold", DEFAULT_Q_HARD_THRESHOLD)

    frozen = FrozenConfig(
        method_id=method_id,
        network_depth=final_params.get("network_depth"),
        network_width=final_params.get("network_width"),
        learning_rate=final_params.get("learning_rate"),
        l2_penalty=final_params.get("l2_penalty"),
        batch_size=final_params.get("batch_size"),
        training_objective=stat_kind,
        kernel_bandwidth_multiplier=final_params.get("kernel_bandwidth_multiplier", 1.0),
        q_hard_threshold=q_hard_threshold,
        n_splits=n_splits,
        max_epochs=final_params.get("n_epochs"),
        patience=final_params.get("patience"),
        early_stopping_min_delta=final_params.get("early_stopping_min_delta"),
        scheduler_factor=final_params.get("scheduler_factor"),
        scheduler_patience=final_params.get("scheduler_patience"),
        scheduler_min_lr=final_params.get("scheduler_min_lr"),
        pilot_select_data_seed=pilot_select_data_seed,
        split_seed=seeds.split_seed,
        tuning_seed=seeds.tuning_seed,
        round1_additional_trials=n_trials,
        round1_total_trials=existing_trial_count + n_trials,
        round1_top_k=top_k,
        round2_init_seeds=list(round2_init_seeds),
        selected_round1_trial_number=selected["round1_trial_number"],
        pure_min_round1_trial_number=pure_min["round1_trial_number"],
    )
    return frozen, round1_table, round2_table


# ---------------------------------------------------------------------------
# Safe end-to-end orchestration: Round 1 -> Round 2 -> one-SE provisional
# candidate -> pilot-confirm -> candidate saved either way, confirm_status
# recorded inside it (精简与config管理.md section 4.1). HPO never writes to
# configs/sgd_frozen/; scripts/run_pilot_protocol.py loops over `--methods`
# itself and calls run_and_confirm_method per method.
# ---------------------------------------------------------------------------

def run_and_confirm_method(
    method_id: str,
    pilot_select_dataset,
    pilot_select_data_seed: int,
    pilot_confirm_dataset,
    data_configs: Dict[str, Any],
    base_train_params: Dict[str, Any],
    search_space: Dict[str, Any],
    n_splits: int,
    n_trials: int,
    base_seed: int,
    top_k: int = 5,
    round2_init_seeds: Tuple[int, ...] = DEFAULT_ROUND2_INIT_SEEDS,
    scenario: Optional[int] = None,
    storage_dir: Optional[Path] = None,
    pilot_select_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    The full, safe pipeline for one method: Round 1 + Round 2 + one-SE
    selection (run_hpo_study_for_method) -> pilot-confirm
    (evaluate_pilot_confirm) -> explicit pass/fail judgment
    (decide_pilot_confirm). The returned `frozen`'s `status` is "confirmed"
    only if pilot-confirm passed, "failed" otherwise; `confirm_status`/
    `confirm_failure_reasons` mirror decide_pilot_confirm()'s own
    "passed"/"failed" verdict. `save_pipeline_artifacts` always saves this
    candidate to `candidate_config_path` regardless of which it is --
    nothing here ever decides to write to configs/sgd_frozen/.

    Round 1 stopping with no finite-valued trial (pilot HPO fix round 2
    item 4) propagates as a plain exception from run_hpo_study_for_method --
    there is no provisional candidate at all in that case, so this function
    does not catch it; the caller (e.g. the script's per-method loop)
    decides whether to skip this method and continue with the others.
    """
    provisional, round1_table, round2_table = run_hpo_study_for_method(
        method_id=method_id, pilot_select_dataset=pilot_select_dataset,
        pilot_select_data_seed=pilot_select_data_seed, data_configs=data_configs,
        base_train_params=base_train_params, search_space=search_space,
        n_splits=n_splits, n_trials=n_trials, base_seed=base_seed,
        top_k=top_k, round2_init_seeds=round2_init_seeds, storage_dir=storage_dir,
        pilot_select_sha256=pilot_select_sha256,
    )

    try:
        report = evaluate_pilot_confirm(
            frozen_config=provisional, pilot_confirm_dataset=pilot_confirm_dataset,
            data_configs=data_configs, scenario=scenario,
        )
    except Exception as e:
        report = {"method_id": method_id, "exception": str(e)}

    decision = decide_pilot_confirm(report)
    frozen = replace(
        provisional,
        status="confirmed" if decision["status"] == "passed" else "failed",
        confirm_status=decision["status"],
        confirm_failure_reasons=decision["failure_reasons"],
    )

    return {
        "method_id": method_id,
        "frozen": frozen,
        "round1_table": round1_table,
        "round2_table": round2_table,
        "confirm_report": {**report, **decision},
    }


def save_pipeline_artifacts(
    pipeline_results: Dict[str, Dict[str, Any]],
    results_dir: Optional[Path] = None,
) -> None:
    """
    Persist every method's Round 1/Round 2 tables, pilot-confirm report,
    and candidate config -- all under `results_dir`, unconditionally,
    regardless of whether that method's pilot-confirm passed or failed
    (精简与config管理.md section 4.1). There is no `config_dir` parameter:
    HPO never writes to configs/sgd_frozen/ itself; a human decides which
    candidate (if any) to copy there by hand (section 4.3).
    """
    results_dir = Path(results_dir) if results_dir is not None else tune_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    for method_id, result in pipeline_results.items():
        frozen = result["frozen"]
        result["round1_table"].to_csv(round1_trials_path(method_id, results_dir), index=False)
        result["round2_table"].to_csv(round2_finalists_path(method_id, results_dir), index=False)
        with open(pilot_confirm_report_path(method_id, results_dir), "w") as f:
            json.dump(result["confirm_report"], f, indent=2, default=str)

        candidate_path = candidate_config_path(method_id, results_dir)
        frozen.save(candidate_path)
        if frozen.confirm_status == "passed":
            print(f"[{method_id}] pilot-confirm PASSED -> candidate saved to {candidate_path}. "
                  f"Nothing was written to {frozen_config_dir()} -- copy it there yourself "
                  f"once you've checked it (plan section 4.3).")
        else:
            print(f"[{method_id}] pilot-confirm FAILED "
                  f"(failure_reasons={frozen.confirm_failure_reasons}); "
                  f"candidate saved to {candidate_path} for inspection, not promoted anywhere.")


# ---------------------------------------------------------------------------
# Plan 6.1 / hpo_selection_plan.md section 9: pilot-confirm evaluation
# ---------------------------------------------------------------------------

def _summarize_fold_diagnostics(fold_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Plan 6.1 / hpo_selection_plan.md section 9's pilot-confirm checklist:
    "held-out 矩风险为有限值" / "所有折都恢复了最佳模型" / "训练时间处于可执行范围".
    """
    risks = [d["moment_risk"] for d in fold_diagnostics]
    sizes = [d["n_eval"] for d in fold_diagnostics]
    seconds = [d["train_seconds"] for d in fold_diagnostics]
    best_epochs = [d["best_epoch"] for d in fold_diagnostics]
    return {
        "mean_moment_risk": weighted_fold_mean(risks, sizes),
        "moment_risk_all_finite": all(math.isfinite(r) for r in risks),
        "all_folds_restored_best_checkpoint": all(
            d["stop_reason"] != "no_valid_checkpoint" for d in fold_diagnostics
        ),
        "any_fold_hit_max_epochs": any(d["stop_reason"] == "max_epochs" for d in fold_diagnostics),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "mean_train_seconds": float(np.mean(seconds)),
        "max_train_seconds": float(np.max(seconds)),
        "total_train_seconds": float(np.sum(seconds)),
    }


def evaluate_pilot_confirm(
    frozen_config: FrozenConfig,
    pilot_confirm_dataset,
    data_configs: Dict[str, Any],
    scenario: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run ONE full cross-fitting pass of `frozen_config` on pilot_confirm
    data, which never participated in this method's HPO. Returns
    diagnostics the caller uses to decide pass/freeze vs fail/back-to-HPO;
    this function itself makes no pass/fail judgment (plan: "pilot-confirm
    只有两种结论...不能静默挑第二名或反复查看 confirm 数据").

    Reports (plan 6.1 / hpo_selection_plan.md section 9, plus the fixed
    hard-failure fields 精简与config管理.md section 6.5 needs --
    `ate_all_finite`/`predictions_all_finite`/(Q only)`q_max_abs`+
    `q_hard_threshold` -- decide_pilot_confirm() reads from this report):
    - point estimate and per-fold estimates, and whether all of them are
      finite;
    - per-fold held-out moment risk (same unified V-statistic scorer HPO
      uses) plus a sample-size-weighted mean across folds, and whether
      every fold actually restored a valid best checkpoint;
    - per-fold training time (a diagnostic/time-table figure only -- it no
      longer gates pass/fail);
    - whether the final OOF H/Q predictions contain any NaN/Inf, and (Q
      only) extreme-value diagnostics plus the max |prediction| compared
      against this method's own `q_hard_threshold`;
    - (scenario 1 only -- the only DGP this repo has a verified closed-form
      true bridge for, see _true_h0_scenario1/_true_q0_scenario1) the true
      bridge MSE, reported for diagnosis ONLY, never used to reselect
      hyperparameters (plan: "不据此更换参数").
    """
    model_kind, _ = frozen_config.method_id.split("-")
    resolved_train_params = frozen_config.to_train_params(model_kind)
    validate_frozen_config(resolved_train_params)

    set_random_seed(frozen_config.tuning_seed)

    if model_kind == "Q":
        from src.models.NMMRq.nmmr_q_experiments import _run_cross_fitting as run_cf
        ate_estimate, ate_values, q_preds, fold_diagnostics = run_cf(
            data_name="sgd", dataset=pilot_confirm_dataset, n_splits=frozen_config.n_splits,
            train_params=resolved_train_params, data_configs=data_configs,
            random_seed=frozen_config.split_seed, log_folder=None, tag="pilot_confirm",
        )
        report: Dict[str, Any] = {
            "method_id": frozen_config.method_id,
            "ate_estimate": ate_estimate.item(),
            "per_fold_ate": ate_values,
            "ate_all_finite": bool(math.isfinite(ate_estimate.item()) and all(math.isfinite(v) for v in ate_values)),
            "n_folds": frozen_config.n_splits,
            "fold_diagnostics": fold_diagnostics,
            **_summarize_fold_diagnostics(fold_diagnostics),
        }
        a_labels = pilot_confirm_dataset.A.cpu().numpy().reshape(-1)
        q_flat = q_preds.reshape(-1)
        report["predictions_all_finite"] = bool(np.isfinite(q_flat).all())
        report["q_max_abs"] = float(np.max(np.abs(q_flat))) if q_flat.size else None
        report["q_hard_threshold"] = frozen_config.q_hard_threshold
        for level, mask_val in ((0, 0.0), (1, 1.0)):
            mask = a_labels == mask_val
            q_level = q_flat[mask]
            if q_level.size == 0:
                continue
            report[f"q_a{level}_mean"] = float(q_level.mean())
            report[f"q_a{level}_sd"] = float(q_level.std())
            report[f"q_a{level}_min"] = float(q_level.min())
            report[f"q_a{level}_max"] = float(q_level.max())
            for pct in (50, 90, 95, 99):
                report[f"q_a{level}_p{pct}"] = float(np.percentile(q_level, pct))
            report[f"q_a{level}_prop_gt10"] = float((q_level > 10).mean())
            report[f"q_a{level}_prop_gt50"] = float((q_level > 50).mean())
            report[f"q_a{level}_prop_gt100"] = float((q_level > 100).mean())
        if scenario == 1:
            true_q = _true_q0_scenario1(
                pilot_confirm_dataset.Z.cpu().numpy().reshape(-1),
                pilot_confirm_dataset.X.cpu().numpy(),
                a_labels,
            )
            report["q_true_bridge_mse"] = float(np.mean((q_flat - true_q) ** 2))
    else:
        from src.models.NMMRh.nmmr_h_experiments import _run_cross_fitting as run_cf
        ate_estimate, ate_values, h_preds, fold_diagnostics = run_cf(
            data_name="sgd", dataset=pilot_confirm_dataset, n_splits=frozen_config.n_splits,
            train_params=resolved_train_params, data_configs=data_configs,
            random_seed=frozen_config.split_seed, log_folder=None, tag="pilot_confirm",
        )
        report = {
            "method_id": frozen_config.method_id,
            "ate_estimate": ate_estimate.item(),
            "per_fold_ate": ate_values,
            "ate_all_finite": bool(math.isfinite(ate_estimate.item()) and all(math.isfinite(v) for v in ate_values)),
            "n_folds": frozen_config.n_splits,
            "fold_diagnostics": fold_diagnostics,
            **_summarize_fold_diagnostics(fold_diagnostics),
        }
        report["predictions_all_finite"] = bool(
            np.isfinite(h_preds["h_fact"]).all()
            and np.isfinite(h_preds["h_1"]).all()
            and np.isfinite(h_preds["h_0"]).all()
        )
        if scenario == 1:
            a_labels = pilot_confirm_dataset.A.cpu().numpy().reshape(-1)
            true_h = _true_h0_scenario1(
                pilot_confirm_dataset.W.cpu().numpy().reshape(-1),
                a_labels,
                pilot_confirm_dataset.X.cpu().numpy(),
            )
            report["h_true_bridge_mse"] = float(np.mean((h_preds["h_fact"].reshape(-1) - true_h) ** 2))

    return report


# ---------------------------------------------------------------------------
# Scenario-1 analytical true bridge (for pilot-confirm diagnostics only --
# plan: "不据此更换参数"). Matches src/data/ate/sgd_pv.py's scenario=1 DGP and
# plan/experiment_1/实验计划.md section 4.1 / plan.md section 3's closed-form
# derivation; sanity-checked against the moment condition in
# local_tests/unit/test_true_bridge_scenario1.py.
# ---------------------------------------------------------------------------

def _true_h0_scenario1(w: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """h_0(w, a, x) = 2 + 2a + 4w + 0.25*x1 + 0.25*x2."""
    s = x[:, 0] + x[:, 1]
    return 2 + 2 * a + 4 * w + 0.25 * s


def _true_q0_scenario1(z: np.ndarray, x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """q_{0,0}(z,x) = 1+exp(0.5z-0.25S-0.25); q_{0,1}(z,x) = 1+exp(-0.5z+0.25S+0.125)."""
    s = x[:, 0] + x[:, 1]
    q0 = 1 + np.exp(0.5 * z - 0.25 * s - 0.25)
    q1 = 1 + np.exp(-0.5 * z + 0.25 * s + 0.125)
    return np.where(a > 0.5, q1, q0)
