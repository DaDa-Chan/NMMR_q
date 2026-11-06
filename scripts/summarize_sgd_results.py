"""
CLI entry for read-only P0/E1/E2 SGD result summarization
(plan/p0_e1_e2/NEXT_IMPLEMENTATION_PROMPT.md "实现 P0/E1/E2 汇总程序"). This
script only parses `--results_dir`, calls
`src.experiments.reporting.summarize_sgd_results`, and prints where the
three summary files were written -- the actual summarization logic lives
in src/experiments/reporting.py, never here, and nothing here trains a
model or runs Monte Carlo.

    python scripts/summarize_sgd_results.py --results_dir results/p0_s1_v1

Safe to run repeatedly on the same `--results_dir`: each call
overwrites e1/bridge_summary.csv, e2/effect_summary.csv, and
common/time_summary.csv in place (never appends), and never touches
run_config.json, the original per-repetition CSVs, or any NPZ prediction
file. If any planned (scenario,n,method,variant) group's results are
incomplete, it refuses and prints why instead of guessing -- delete that
results directory and rerun the whole experiment (this protocol has no
resume support).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.reporting import summarize_sgd_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True, help="已完成的 P0/E1/E2 结果目录，例如 results/p0_s1_v1")
    args = parser.parse_args()

    paths = summarize_sgd_results(Path(args.results_dir))
    print("汇总完成，已生成/覆盖：")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
