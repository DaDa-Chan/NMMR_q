"""
End-to-end driver for the HPO pilot protocol (training_and_hpo_plan.md
sections 5.1, 5.2, 6, the more detailed two-round version in
hpo_selection_plan.md sections 3-10, and the candidate/frozen-config
separation in 精简与config管理.md): generate pilot_select/pilot_confirm
data, run each selected method through the two-round selection (Round 1
Optuna screening -> Round 2 multi-seed re-evaluation of the finalists ->
one-SE provisional pick), evaluate that candidate on pilot_confirm, and
save it to `--results_dir/candidates/` either way -- this script NEVER
writes to configs/sgd_frozen/ itself. A `confirm_status` field
("passed"/"failed") inside the saved candidate tells you which one it was;
promoting a passed candidate to configs/sgd_frozen/ for the formal
experiment is a manual copy you do yourself once you've checked it (see
README.md's "HPO FrozenConfig 与正式实验的当前接口" section).

Output goes under `--results_dir` (default tune_results/) plus
`--data_dir` (default data/pilot/) -- see src/models/hpo_protocol.py's
module docstring for the exact layout. Running this script is "doing real
pilot work," even at small `n` -- unlike the automated tests under
local_tests/, which always use their own tempdir and never touch these
folders.

Pilot data is kept under --data_dir/{dgp_version}/scenario_{scenario}/
seed_{base_seed}/, one file per (role, sample size): select_n{n}.csv,
confirm_n{n}.csv, plus a shared manifest.json recording each file's role,
n, data_seed, and SHA-256. Requesting a different --n_pilot_select (e.g.
10000 after a prior run used 5000) adds select_n10000.csv alongside
select_n5000.csv -- it never overwrites it. Each (role, n) that already
exists and matches its manifest record is reused as-is by default; pass
--force-regenerate to explicitly overwrite ONLY the exact (--n_pilot_select,
--n_pilot_confirm) pair requested in this invocation, never any other size.

Defaults here are deliberately small (hundreds of samples, a handful of
trials/epochs, 2 round-2 seeds) so this can be run as a correctness/smoke
check of the whole pipeline without burning hours of compute. For an
actual production pilot, override --n_pilot_select/--n_pilot_confirm/
--n_trials/--n_epochs/--round2_init_seeds with the plan's real values
(5000/5000/30-50/300/"11,22,33") -- that run will take much longer and
should be started deliberately, not as a side effect of a quick check.

Usage:
    python scripts/run_pilot_protocol.py
    python scripts/run_pilot_protocol.py --methods Q-U
    python scripts/run_pilot_protocol.py --methods Q-U,Q-V
    python scripts/run_pilot_protocol.py --n_pilot_select 5000 --n_pilot_confirm 5000 \
        --n_trials 30 --n_epochs 300 --round2_init_seeds 11,22,33
    python scripts/run_pilot_protocol.py --force-regenerate
"""
import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from src.models import hpo_protocol as hp
from src.models.experiment_common import _parse_optuna_space


def _load_method_config(method_id: str):
    """Load this method's real config file under configs/sgd/, the same
    files production runs would use (already updated to plan 5.6's search
    space).
    Returns (data_configs, train_params, raw_bytes) -- `raw_bytes` is the
    file's exact, unmodified bytes (parsed via `json.loads`, never
    round-tripped through `json.dump`), saved verbatim as this run's search
    config snapshot (精简与config管理.md section 4.2) via
    hp.ensure_search_config_snapshot, before any CLI override
    (--n_epochs/--n_splits/--patience) is applied to `train_params`."""
    model_kind, stat_kind = method_id.split("-")
    model_dir = "nmmr_q" if model_kind == "Q" else "nmmr_h"
    config_path = PROJECT_ROOT / "configs" / "sgd" / model_dir / f"{model_dir}_{stat_kind.lower()}_sgd.json"
    raw_bytes = config_path.read_bytes()
    cfg = json.loads(raw_bytes)
    return cfg["data_configs"]["sgd"], cfg["train_params"], raw_bytes


def _parse_methods(raw: str):
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    return hp.validate_method_ids(methods)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods", type=str, default=",".join(hp.METHOD_IDS),
        help=f"Comma-separated subset of {hp.METHOD_IDS} to run (default: all four). "
             f"e.g. --methods Q-U or --methods Q-U,Q-V",
    )
    parser.add_argument("--n_pilot_select", type=int, default=300)
    parser.add_argument("--n_pilot_confirm", type=int, default=300)
    parser.add_argument(
        "--n_trials", type=int, default=3,
        help="Number of additional Optuna trials to run in this invocation. "
             "Existing trials are preserved.",
    )
    parser.add_argument("--top_k", type=int, default=2, help="Round 1 finalists carried into Round 2")
    parser.add_argument("--round2_init_seeds", type=str, default="11,22",
                         help="Comma-separated init seeds for Round 2 (plan default: 11,22,33)")
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=20261103)
    parser.add_argument("--data_dir", type=str, default=str(hp.pilot_data_dir()))
    parser.add_argument("--results_dir", type=str, default=str(hp.tune_results_dir()))
    parser.add_argument(
        "--force-regenerate", action="store_true", dest="force_regenerate",
        help="Overwrite ONLY the select_n{n_pilot_select}.csv / confirm_n{n_pilot_confirm}.csv "
             "pair for this invocation's sample sizes (default: reuse them as-is if they already "
             "match the manifest). Never touches any other sample size's file.",
    )
    args = parser.parse_args()

    method_ids = _parse_methods(args.methods)
    round2_init_seeds = tuple(int(s) for s in args.round2_init_seeds.split(","))
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    print(f"=== Methods selected: {method_ids} ===")

    print(f"\n=== Step 1/3: pilot_select (n={args.n_pilot_select}) and pilot_confirm "
          f"(n={args.n_pilot_confirm}) -> {data_dir} "
          f"(force_regenerate={args.force_regenerate}) ===")
    pilot = hp.generate_pilot_datasets(
        base_seed=args.base_seed, n_pilot_select=args.n_pilot_select,
        n_pilot_confirm=args.n_pilot_confirm, scenario=args.scenario, save_dir=data_dir,
        force_regenerate=args.force_regenerate,
    )
    print(f"select_data_seed={pilot.select_data_seed}, confirm_data_seed={pilot.confirm_data_seed}")

    print(f"\n=== Step 2/3: Round 1 ({args.n_trials} trials, top_k={args.top_k}) -> Round 2 "
          f"(seeds={round2_init_seeds}) -> one-SE candidate -> pilot-confirm -> candidate saved "
          f"to {results_dir}/candidates/ either way (confirm_status records passed/failed), "
          f"for {method_ids} ===")

    pipeline_results = {}
    # 精简与config管理.md Phase A item 2: Round 1 and Round 2 total failure
    # are DISTINCT, named exception types -- never both reported as the same
    # generic "Round 1 failed". Any OTHER exception is not caught here at
    # all: it propagates with its own type and message instead of being
    # relabeled as a Round 1 failure.
    round1_failed_methods = []
    round2_failed_methods = []
    for method_id in method_ids:
        data_configs, train_params, raw_cfg_bytes = _load_method_config(method_id)

        # 精简与config管理.md section 4.2: a human-readable, byte-exact copy
        # of the search config this method used THIS run. Write-once: reused
        # as-is if it already matches (an ordinary resume), but if this
        # results_dir's snapshot was taken from a DIFFERENT config, this
        # raises immediately -- no HPO runs for this method, and the old
        # snapshot is left completely untouched. Use a new --results_dir for
        # a genuinely changed search space instead of reusing this one.
        snapshot_path = hp.search_config_snapshot_path(
            results_dir, method_id=method_id if len(method_ids) > 1 else None,
        )
        hp.ensure_search_config_snapshot(raw_cfg_bytes, snapshot_path)

        base_params, search_space = _parse_optuna_space(copy.deepcopy(train_params))
        # Override only the knobs that control run length, so this stays a
        # real dry run of the actual search space, not a toy one.
        base_params["n_epochs"] = args.n_epochs
        base_params["n_splits"] = args.n_splits
        base_params["patience"] = args.patience

        print(f"\n--- {method_id} (search config snapshot -> {snapshot_path}) ---")
        try:
            result = hp.run_and_confirm_method(
                method_id=method_id,
                pilot_select_dataset=pilot.select_dataset,
                pilot_select_data_seed=pilot.select_data_seed,
                pilot_confirm_dataset=pilot.confirm_dataset,
                data_configs=data_configs,
                base_train_params=base_params,
                search_space=search_space,
                n_splits=args.n_splits, n_trials=args.n_trials, base_seed=args.base_seed,
                top_k=args.top_k, round2_init_seeds=round2_init_seeds,
                scenario=args.scenario, storage_dir=results_dir,
                pilot_select_sha256=pilot.select_sha256,
            )
        except hp.NoFiniteRound1TrialError as e:
            # No provisional candidate exists at all for this method, so
            # there is nothing for save_pipeline_artifacts to write. Skip
            # it and keep going with the other requested methods.
            print(f"[{method_id}] round1_failed: {e}")
            round1_failed_methods.append(method_id)
            continue
        except hp.NoFiniteRound2CandidateError as e:
            print(f"[{method_id}] round2_failed: {e}")
            round2_failed_methods.append(method_id)
            continue
        pipeline_results[method_id] = result

    print(f"\n=== Step 3/3: saving Round 1/2 tables, pilot-confirm reports, and every "
          f"method's candidate (confirm_status=passed|failed either way) -> {results_dir} ===")
    hp.save_pipeline_artifacts(pipeline_results, results_dir=results_dir)

    print("\n=== Summary ===")
    for method_id, result in pipeline_results.items():
        frozen = result["frozen"]
        report = result["confirm_report"]
        print(
            f"{method_id}: confirm_status={frozen.confirm_status}, "
            f"failure_reasons={frozen.confirm_failure_reasons}, "
            f"ate_estimate={report.get('ate_estimate')}, "
            f"mean_moment_risk={report.get('mean_moment_risk')}, "
            f"candidate={hp.candidate_config_path(method_id, results_dir)}"
        )
    for method_id in round1_failed_methods:
        print(f"{method_id}: round1_failed (no finite Round 1 trial; no candidate at all)")
    for method_id in round2_failed_methods:
        print(f"{method_id}: round2_failed (no finite Round 2 candidate; no candidate at all)")

    print(f"\nAll artifacts written to {data_dir}, {results_dir}. "
          f"Nothing was written to {hp.frozen_config_dir()} -- "
          f"promote a passed candidate there yourself once you've checked it.")


if __name__ == "__main__":
    main()
