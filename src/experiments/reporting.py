"""
Read-only summarization of a completed P0/E1/E2 SGD results directory
(plan/p0_e1_e2/NEXT_IMPLEMENTATION_PROMPT.md "实现 P0/E1/E2 汇总程序",
plan/p0_e1_e2/05_reporting.md section 3, plan/p0_e1_e2/04_e2_ate_comparison.md
section 5). This module never trains a model, never runs Monte Carlo, and
never modifies `run_config.json`, the original per-repetition CSVs, or any
NPZ prediction file -- it only reads

    run_config.json
    e1/bridge_results.csv
    e2/effect_results.csv
    common/failures.csv        (optional -- missing means zero failures)

and writes/overwrites, in place, with no accumulation across runs:

    e1/bridge_summary.csv
    e2/effect_summary.csv
    common/time_summary.csv

Every summary groups by (run_stage, scenario, n, method, variant); pandas
reads an empty-string `variant` field back as NaN, which is coerced back
to "" here so oracle/parametric_proximal/NMMR (which have no variant)
survive the groupby instead of silently disappearing. Before writing
anything, every planned group's `successes + failures` is checked against
`run_config.json`'s own `repetitions` -- a mismatch raises
`IncompleteResultsError` rather than guessing at, retrying, or recomputing
the missing rows (that results directory must be deleted and the whole
experiment rerun, same as this protocol's existing no-resume rule).
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .scenarios import get_scenario
from .sgd_experiment_protocol import LS_VARIANTS, _VARIANT_EXPANDED_METHODS

GROUP_KEYS: Tuple[str, ...] = ("run_stage", "scenario", "n", "method", "variant")
# Fixed estimator order (NEXT_IMPLEMENTATION_PROMPT.md section 6: "effect
# summary 再按 POR → PIPW → PDR") -- never alphabetical.
ESTIMATORS: Tuple[str, ...] = ("POR", "PIPW", "PDR")
_ESTIMATOR_TAU_COLUMNS: Dict[str, str] = {"POR": "tau_por", "PIPW": "tau_pipw", "PDR": "tau_pdr"}

# The bridge metrics averaged as-is across successful repetitions (section
# 3) -- never a new kernel moment risk, never a threshold-based filter.
_BRIDGE_METRIC_COLUMNS: Tuple[str, ...] = (
    "h_mse_a0", "h_mse_a1", "h_mse",
    "h_mae_a0", "h_mae_a1", "h_mae",
    "q_mse_a0", "q_mse_a1", "q_mse",
    "q_mae_a0", "q_mae_a1", "q_mae",
    "h_moment_l2", "h_moment_linf",
    "q_moment_l2", "q_moment_linf",
    "q_min", "q_p50", "q_p90", "q_p95", "q_p99", "q_max", "q_negative_rate",
)
_TIME_MEAN_COLUMNS: Tuple[str, ...] = ("h_train_seconds", "q_train_seconds", "evaluation_seconds")


class IncompleteResultsError(RuntimeError):
    """A (run_stage,scenario,n,method,variant) group's `successes +
    failures` does not equal `run_config.json`'s planned `repetitions` --
    this results directory is incomplete (an interrupted/still-running
    experiment, or one pointed at the wrong --results_dir). Delete it and
    rerun the whole experiment; `summarize_sgd_results` never repairs or
    guesses at missing rows."""


def _normalize_variant(series: pd.Series) -> pd.Series:
    """An empty-string `variant` round-trips through CSV as NaN -- always
    coerce back to "" so oracle/parametric_proximal/NMMR rows (which have
    no variant) are not silently dropped by a later groupby/equality check."""
    return series.fillna("").astype(str)


def _read_csv_or_empty(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    df = pd.read_csv(path)
    if "variant" in df.columns:
        df["variant"] = _normalize_variant(df["variant"])
    return df


def _planned_method_variants(methods: Sequence[str]) -> List[Tuple[str, str]]:
    """Reconstructs every (method, variant) run spec EXACTLY the way
    `run_sgd_protocol` itself expands `--methods` -- reusing its own
    `LS_VARIANTS`/`_VARIANT_EXPANDED_METHODS` rather than a second,
    independently-maintained copy of that rule."""
    specs: List[Tuple[str, str]] = []
    for method in methods:
        if method in _VARIANT_EXPANDED_METHODS:
            specs.extend((method, variant) for variant in LS_VARIANTS)
        else:
            specs.append((method, ""))
    return specs


def _planned_groups(run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every (run_stage,scenario,n,method,variant) group this run_config
    planned to produce, in the stable order NEXT_IMPLEMENTATION_PROMPT.md
    section 6 asks for: run_stage -> scenario -> n -> method -> variant,
    taken straight from run_config.json's own list order (never re-sorted)."""
    run_stage = run_config["run_stage"]
    scenario_ids = run_config["scenarios"]
    sample_sizes = run_config["sample_sizes"]
    method_variants = _planned_method_variants(run_config["methods"])
    return [
        {"run_stage": run_stage, "scenario": sid, "n": n, "method": m, "variant": v}
        for sid in scenario_ids
        for n in sample_sizes
        for (m, v) in method_variants
    ]


def _select(df: pd.DataFrame, group: Dict[str, Any]) -> pd.DataFrame:
    if len(df) == 0:
        return df.iloc[0:0]
    mask = pd.Series(True, index=df.index)
    for key, value in group.items():
        if key not in df.columns:
            return df.iloc[0:0]
        mask &= (df[key] == value)
    return df[mask]


def _check_completeness(
    groups: List[Dict[str, Any]], effect: pd.DataFrame, failures: pd.DataFrame, planned_repetitions: int,
) -> None:
    mismatches = []
    for group in groups:
        n_success = len(_select(effect, group))
        n_failure = len(_select(failures, group))
        if n_success + n_failure != planned_repetitions:
            mismatches.append({**group, "successes": n_success, "failures": n_failure})
    if mismatches:
        raise IncompleteResultsError(
            f"{len(mismatches)} group(s) have successes+failures != planned repetitions "
            f"({planned_repetitions}); this results directory is incomplete. Delete it and "
            f"rerun the whole experiment instead of summarizing a partial one. "
            f"First mismatch: {mismatches[0]}."
        )


def _group_counts(
    success_df: pd.DataFrame, failures: pd.DataFrame, group: Dict[str, Any], planned_repetitions: int,
) -> Tuple[pd.DataFrame, int, int, int, float]:
    group_success = _select(success_df, group)
    group_failures = _select(failures, group)
    successes = len(group_success)
    n_failures = len(group_failures)
    attempts = planned_repetitions
    failure_rate = (n_failures / attempts) if attempts else float("nan")
    return group_success, attempts, successes, n_failures, failure_rate


def _build_bridge_summary(
    bridge: pd.DataFrame, failures: pd.DataFrame, groups: List[Dict[str, Any]], planned_repetitions: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        group_bridge, attempts, successes, n_failures, failure_rate = _group_counts(
            bridge, failures, group, planned_repetitions,
        )
        row: Dict[str, Any] = {
            **group, "attempts": attempts, "successes": successes, "failures": n_failures,
            "failure_rate": failure_rate,
        }
        for col in _BRIDGE_METRIC_COLUMNS:
            row[col] = float(group_bridge[col].astype(float).mean()) if successes > 0 else float("nan")
        rows.append(row)
    columns = [*GROUP_KEYS, "attempts", "successes", "failures", "failure_rate", *_BRIDGE_METRIC_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def _build_effect_summary(
    effect: pd.DataFrame, failures: pd.DataFrame, groups: List[Dict[str, Any]], planned_repetitions: int,
    scenario_true_ate: Dict[str, float],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        group_effect, attempts, successes, n_failures, failure_rate = _group_counts(
            effect, failures, group, planned_repetitions,
        )
        true_ate = scenario_true_ate[group["scenario"]]

        if successes > 0:
            total_seconds = group_effect["total_seconds"].astype(float).to_numpy()
            median_time = float(np.median(total_seconds))
            p90_time = float(np.percentile(total_seconds, 90))
        else:
            median_time = float("nan")
            p90_time = float("nan")

        for estimator in ESTIMATORS:
            row: Dict[str, Any] = {
                **group, "estimator": estimator, "true_ate": true_ate,
                "attempts": attempts, "successes": successes, "failures": n_failures,
                "failure_rate": failure_rate, "median_time": median_time, "p90_time": p90_time,
            }
            if successes > 0:
                taus = group_effect[_ESTIMATOR_TAU_COLUMNS[estimator]].astype(float).to_numpy()
                mean = float(np.mean(taus))
                bias = mean - true_ate
                rmse = float(np.sqrt(np.mean((taus - true_ate) ** 2)))
                if successes >= 2:
                    sd = float(np.std(taus, ddof=1))
                    mcse_bias = sd / np.sqrt(successes)
                else:
                    sd = float("nan")
                    mcse_bias = float("nan")
                row.update(mean=mean, bias=bias, abs_bias=abs(bias), sd=sd, rmse=rmse, mcse_bias=mcse_bias)
            else:
                row.update(
                    mean=float("nan"), bias=float("nan"), abs_bias=float("nan"),
                    sd=float("nan"), rmse=float("nan"), mcse_bias=float("nan"),
                )
            rows.append(row)

    columns = [
        *GROUP_KEYS, "estimator", "true_ate",
        "attempts", "successes", "failures", "failure_rate",
        "mean", "bias", "abs_bias", "sd", "rmse", "mcse_bias",
        "median_time", "p90_time",
    ]
    return pd.DataFrame(rows, columns=columns)


def _build_time_summary(
    effect: pd.DataFrame, failures: pd.DataFrame, groups: List[Dict[str, Any]], planned_repetitions: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        group_effect, attempts, successes, n_failures, failure_rate = _group_counts(
            effect, failures, group, planned_repetitions,
        )
        row: Dict[str, Any] = {
            **group, "attempts": attempts, "successes": successes, "failures": n_failures,
            "failure_rate": failure_rate,
        }
        if successes > 0:
            for col in _TIME_MEAN_COLUMNS:
                row[f"{col}_mean"] = float(group_effect[col].astype(float).mean())
            total_seconds = group_effect["total_seconds"].astype(float).to_numpy()
            row["total_seconds_mean"] = float(np.mean(total_seconds))
            row["total_seconds_median"] = float(np.median(total_seconds))
            row["total_seconds_p90"] = float(np.percentile(total_seconds, 90))
        else:
            for col in _TIME_MEAN_COLUMNS:
                row[f"{col}_mean"] = float("nan")
            row["total_seconds_mean"] = float("nan")
            row["total_seconds_median"] = float("nan")
            row["total_seconds_p90"] = float("nan")
        rows.append(row)

    columns = [
        *GROUP_KEYS, "attempts", "successes", "failures", "failure_rate",
        "h_train_seconds_mean", "q_train_seconds_mean", "evaluation_seconds_mean",
        "total_seconds_mean", "total_seconds_median", "total_seconds_p90",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_sgd_results(results_dir: Path) -> Dict[str, Path]:
    """
    Reads `results_dir`'s run_config.json + e1/bridge_results.csv +
    e2/effect_results.csv + (optional) common/failures.csv, and
    writes/overwrites e1/bridge_summary.csv, e2/effect_summary.csv,
    common/time_summary.csv in place (no accumulation -- a second call on
    the same directory replaces all three files, never appends). Raises
    `IncompleteResultsError` instead of writing anything if any planned
    (scenario,n,method,variant) group's successes+failures doesn't match
    run_config.json's own `repetitions`. Returns the three output paths.
    """
    results_dir = Path(results_dir)
    run_config = json.loads((results_dir / "run_config.json").read_text())
    planned_repetitions = int(run_config["repetitions"])
    groups = _planned_groups(run_config)
    scenario_true_ate = {sid: get_scenario(sid).true_ate for sid in run_config["scenarios"]}

    bridge = _read_csv_or_empty(
        results_dir / "e1" / "bridge_results.csv",
        ["run_stage", "scenario", "n", "repetition", "method", "variant", *_BRIDGE_METRIC_COLUMNS],
    )
    effect = _read_csv_or_empty(
        results_dir / "e2" / "effect_results.csv",
        ["run_stage", "scenario", "n", "repetition", "method", "variant",
         "tau_por", "tau_pipw", "tau_pdr",
         "h_train_seconds", "q_train_seconds", "evaluation_seconds", "total_seconds"],
    )
    failures = _read_csv_or_empty(
        results_dir / "common" / "failures.csv",
        ["run_stage", "scenario", "n", "repetition", "method", "variant", "failure_stage", "failure_reason"],
    )

    _check_completeness(groups, effect, failures, planned_repetitions)

    bridge_summary = _build_bridge_summary(bridge, failures, groups, planned_repetitions)
    effect_summary = _build_effect_summary(effect, failures, groups, planned_repetitions, scenario_true_ate)
    time_summary = _build_time_summary(effect, failures, groups, planned_repetitions)

    out_paths = {
        "bridge_summary": results_dir / "e1" / "bridge_summary.csv",
        "effect_summary": results_dir / "e2" / "effect_summary.csv",
        "time_summary": results_dir / "common" / "time_summary.csv",
    }
    for path in out_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    bridge_summary.to_csv(out_paths["bridge_summary"], index=False)
    effect_summary.to_csv(out_paths["effect_summary"], index=False)
    time_summary.to_csv(out_paths["time_summary"], index=False)
    return out_paths
