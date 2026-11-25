"""
Stage 1 of the RHC analysis: cross-fitted estimates over repetitions, with the
configurations chosen in stage 0, then the DML median combination.

    python scripts/run_rhc_protocol.py --selected_dir results/rhc_tuning \
        --out_dir results/rhc_main --base_seed 20261127 --repetitions 0-19

    # strict nested sensitivity: one repetition, fold-specific configurations
    python scripts/run_rhc_protocol.py --fold_selected_dirs results/rhc_tuning_strict/fold{1..5} \
        --out_dir results/rhc_strict --base_seed 20261127 --repetitions 0

    # only (re)write the combined table
    python scripts/run_rhc_protocol.py --out_dir results/rhc_main --combine_only

Writes results.csv, predictions/ (with per-method fold diagnostics), run_config.json and
combined.csv in --out_dir. Rerunning the same command resumes.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ate.rhc_protocol_data import build_rhc_frame  # noqa: E402
from src.experiments import rhc_protocol as P  # noqa: E402


def parse_repetitions(text: str):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--selected_dir", type=Path)
    p.add_argument("--fold_selected_dirs", type=Path, nargs="+")
    p.add_argument("--base_seed", type=int)
    p.add_argument("--repetitions", default="0-19")
    p.add_argument("--methods", default=",".join(P.ALL_METHODS))
    p.add_argument("--n_splits", type=int, default=P.DEFAULT_N_SPLITS)
    p.add_argument("--combine_only", action="store_true")
    args = p.parse_args()

    if not args.combine_only:
        if args.base_seed is None:
            p.error("--base_seed is required")
        P.run_rhc_protocol(
            build_rhc_frame(), args.out_dir, args.base_seed, parse_repetitions(args.repetitions),
            [m.strip() for m in args.methods.split(",") if m.strip()], selected_dir=args.selected_dir,
            fold_selected_dirs=args.fold_selected_dirs, n_splits=args.n_splits,
        )
    combined = P.combine_results(pd.read_csv(args.out_dir / "results.csv"))
    combined.to_csv(args.out_dir / "combined.csv", index=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
