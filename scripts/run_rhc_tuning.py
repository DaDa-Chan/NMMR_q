"""
Stage 0 of the RHC analysis: tune H-V, Q-V, H-U, Q-U on a fixed candidate pool.

    # write (or check) the candidate pool file, once, before any tuning
    python scripts/run_rhc_tuning.py --write_pool

    # main analysis: tune on the whole sample
    python scripts/run_rhc_tuning.py --out_dir results/rhc_tuning --base_seed 20261126

    # strict nested sensitivity: tune inside one outer training fold of a
    # stage-1 repetition (its evaluation rows are never seen)
    python scripts/run_rhc_tuning.py --out_dir results/rhc_tuning_strict/fold1 --base_seed 20261126 \
        --outer_base_seed 20261127 --outer_repetition 0 --outer_fold 1

Each task writes <out_dir>/<task>/{run_config.json, units.csv, screen_summary.csv,
reeval_summary.csv, selected.json}. Rerunning the same command resumes.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ate.rhc_protocol_data import RHCFrameDataset, build_rhc_frame  # noqa: E402
from src.experiments import rhc_tuning as T  # noqa: E402
from src.experiments.rhc_protocol import DEFAULT_N_SPLITS, repetition_seeds  # noqa: E402
from src.experiments.sgd_experiment_protocol import build_fold_plan  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write_pool", action="store_true", help="write/check the candidate pool file and exit")
    p.add_argument("--pool", type=Path, default=T.DEFAULT_POOL_PATH)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--base_seed", type=int)
    p.add_argument("--tasks", default=",".join(T.TUNING_TASKS))
    p.add_argument("--outer_base_seed", type=int, help="stage-1 base seed (strict nested only)")
    p.add_argument("--outer_repetition", type=int)
    p.add_argument("--outer_fold", type=int, help="1-based outer fold whose training rows are tuned on")
    p.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--kernel_form", default="joint", choices=["joint", "block_scaled", "product"],
                   help="kernel over (A),(Z),(X) for h and (W),(X) for q; used for training and scoring")
    args = p.parse_args()

    if args.write_pool:
        print(f"{args.pool}: sha256={T.save_candidate_pool(args.pool)}")
        return
    if args.out_dir is None or args.base_seed is None:
        p.error("--out_dir and --base_seed are required")

    frame = build_rhc_frame()
    strict = [args.outer_base_seed, args.outer_repetition, args.outer_fold]
    if any(v is not None for v in strict):
        if any(v is None for v in strict):
            p.error("strict nested mode needs --outer_base_seed, --outer_repetition and --outer_fold")
        seeds = repetition_seeds(args.outer_base_seed, args.outer_repetition)
        plan = build_fold_plan(RHCFrameDataset(frame), args.n_splits, seeds.split_seed)
        train_idx, _ = plan.splits[args.outer_fold - 1]
        frame = frame.iloc[train_idx]
        print(f"strict nested: repetition {args.outer_repetition}, outer fold {args.outer_fold}, "
              f"{len(frame)} training rows (fold plan {plan.fold_plan_hash[:12]})")

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        T.run_tuning_task(task, frame, args.out_dir / task, args.base_seed, pool_path=args.pool,
                          kernel_form=args.kernel_form)


if __name__ == "__main__":
    main()
