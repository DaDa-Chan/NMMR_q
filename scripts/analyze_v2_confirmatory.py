"""
Confirmatory analysis CLI for 实验计划 v2 (S1). Read-only over result
directories; all logic lives in src/experiments/v2_analysis.py.

    python scripts/analyze_v2_confirmatory.py \
        --e2_dirs results/v2_E2_n500_1000 results/v2_E2_n2000_CV results/v2_E2_n2000_U \
                  results/v2_E2_n5000_CV results/v2_E2_n5000_U results/v2_E2_n10000 \
        --e3 16=results/v2_E3_b16 32=results/v2_E3_b32 64=results/v2_E3_b64 \
             128=results/v2_E3_b128 256=results/v2_E3_b256 \
        --out_dir results/v2_analysis

Writes e2_bias_table.csv, e2_slopes.csv, e4_coverage.csv, e1_mechanism.csv,
e3_bias_table.csv, e3_slopes.csv, decisions.json and decisions.md into
--out_dir. --e2_dirs and --e3 are both optional, but at least one is needed.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.scenarios import get_scenario
from src.experiments.v2_analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e2_dirs", nargs="*", default=[], help="E2/E1/E4 result directories (merged)")
    parser.add_argument("--e3", nargs="*", default=[], help="E3 directories as batch_size=dir")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scenario", default="S1")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--skip_e1", action="store_true", help="skip the prediction-based E1 table")
    args = parser.parse_args()

    e3 = {}
    for item in args.e3:
        b, _, d = item.partition("=")
        if not d:
            parser.error(f"--e3 entries must look like 16=results/v2_E3_b16, got {item!r}")
        e3[int(b)] = Path(d)
    if not args.e2_dirs and not e3:
        parser.error("give --e2_dirs and/or --e3")

    spec = get_scenario(args.scenario)
    out = run_analysis([Path(d) for d in args.e2_dirs], e3, Path(args.out_dir), true_ate=spec.true_ate,
                       scenario_spec=spec, n_boot=args.n_boot, seed=args.seed, with_e1=not args.skip_e1)
    for d in out["decisions"]:
        print(f"{d['item']}: {d['result']}")
    print(f"tables written to {out['out_dir']}")


if __name__ == "__main__":
    main()
