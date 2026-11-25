"""SGD experiment entry point (P0/E1/E2/E3 protocol runs)."""
import argparse
from pathlib import Path
import sys

# --- 路径设置 ---
current_dir = Path(__file__).parent.resolve()
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.experiments.sgd_experiment_protocol import run_sgd_protocol, ALLOWED_METHODS


def run_sgd_protocol_experiment(
    run_stage: str,
    scenarios: list,
    methods: list,
    sample_sizes: list,
    repetitions: int,
    test_size: int,
    n_splits: int,
    base_seed: int,
    h_u_frozen_config: str,
    h_v_frozen_config: str,
    q_u_frozen_config: str,
    q_v_frozen_config: str,
    output_dir: str,
):
    """
    SGD 的 P0/E1/E2 入口：只读取已通过 confirm 的 FrozenConfig，不运行 Optuna。
    场景、分折、方法调度和结果写入都在 src/experiments/sgd_experiment_protocol.py。

    不支持断点恢复：`output_dir` 必须是全新的空目录；如果某次运行中断或
    失败，删除该不完整目录后重新运行（见 README "SGD P0/E1/E2 实验运行规则"）。
    """
    print("开始 SGD P0/E1/E2 protocol 实验")
    print(f"run_stage={run_stage}, scenarios={scenarios}, methods={methods}")
    print(f"sample_sizes={sample_sizes}, repetitions={repetitions}, test_size={test_size}, n_splits={n_splits}")

    frozen_config_paths = {
        "H-U": Path(h_u_frozen_config),
        "H-V": Path(h_v_frozen_config),
        "Q-U": Path(q_u_frozen_config),
        "Q-V": Path(q_v_frozen_config),
    }
    summary = run_sgd_protocol(
        run_stage=run_stage,
        scenario_ids=scenarios,
        methods=methods,
        sample_sizes=sample_sizes,
        repetitions=repetitions,
        test_size=test_size,
        n_splits=n_splits,
        base_seed=base_seed,
        frozen_config_paths=frozen_config_paths,
        output_dir=Path(output_dir),
    )
    print(f"完成: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='sgd',
        choices=['sgd'],
        help="只支持 'sgd'（保留该参数以兼容已有运行命令）；RHC 使用单独的入口"
    )
    parser.add_argument('--run_stage', type=str, default='P0', help="P0/E1/E2 (仅 SGD)")
    parser.add_argument('--scenarios', type=str, default='S1', help="逗号分隔的场景 id, 如 'S1' (仅 SGD)")
    parser.add_argument(
        '--methods', type=str, default='nmmr_u,nmmr_v',
        help=f"逗号分隔的方法, 可选 {ALLOWED_METHODS} (仅 SGD)"
    )
    parser.add_argument('--sample_sizes', type=str, default=None, help="逗号分隔的样本量, 如 '500,1000' (仅 SGD)")
    parser.add_argument('--repetitions', type=int, default=None, help="重复数 (仅 SGD)")
    parser.add_argument('--test_size', type=int, default=None, help="独立测试集样本量 (仅 SGD)")
    parser.add_argument('--n_splits', type=int, default=5, help="cross-fitting 折数 (仅 SGD)")
    parser.add_argument('--base_seed', type=int, default=None, help="派生所有 repetition seed 的基础种子 (仅 SGD)")
    parser.add_argument('--h_u_frozen_config', type=str, default='configs/sgd_frozen/h_u_frozen_config.json')
    parser.add_argument('--h_v_frozen_config', type=str, default='configs/sgd_frozen/h_v_frozen_config.json')
    parser.add_argument('--q_u_frozen_config', type=str, default='configs/sgd_frozen/q_u_frozen_config.json')
    parser.add_argument('--q_v_frozen_config', type=str, default='configs/sgd_frozen/q_v_frozen_config.json')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="结果输出目录, 如 results/p0_s1_v1 (仅 SGD)。必须是全新的空目录: "
             "不支持断点恢复, 运行中断或失败后请删除该目录重新运行。"
    )

    args = parser.parse_args()

    missing = [
        name for name in ('sample_sizes', 'repetitions', 'test_size', 'base_seed', 'output_dir')
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            f"--dataset_name sgd 需要显式指定: {', '.join('--' + m for m in missing)}"
        )
    run_sgd_protocol_experiment(
        run_stage=args.run_stage,
        scenarios=[s.strip() for s in args.scenarios.split(',') if s.strip()],
        methods=[m.strip() for m in args.methods.split(',') if m.strip()],
        sample_sizes=[int(x) for x in args.sample_sizes.split(',') if x.strip()],
        repetitions=args.repetitions,
        test_size=args.test_size,
        n_splits=args.n_splits,
        base_seed=args.base_seed,
        h_u_frozen_config=args.h_u_frozen_config,
        h_v_frozen_config=args.h_v_frozen_config,
        q_u_frozen_config=args.q_u_frozen_config,
        q_v_frozen_config=args.q_v_frozen_config,
        output_dir=args.output_dir,
    )
