import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# --- 路径设置 ---
current_dir = Path(__file__).parent.resolve()
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.ate.data_class import RHCDataset, MergedDataset, BootstrapDataset
from src.models.NMMRq.nmmr_q_experiments import NMMR_Q_experiment
from src.models.NMMRh.nmmr_h_experiments import NMMR_H_experiment
from src.utils import get_device, set_random_seed, make_seed_bundle
from src.experiments.sgd_experiment_protocol import run_sgd_protocol, ALLOWED_METHODS

def load_config_content(config_path: Path, dataset_name: str):
    """读取 JSON 配置文件内容"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    
    # 提取 data_configs 和 train_params
    if dataset_name not in config_json['data_configs']:
        keys = list(config_json['data_configs'].keys())
        if len(keys) == 1:
            data_configs = config_json['data_configs'][keys[0]]
        else:
            raise KeyError(f"在 {config_path} 中未找到数据集 '{dataset_name}' 的配置。")
    else:
        data_configs = config_json['data_configs'][dataset_name]
        
    train_params = config_json['train_params']
    return data_configs, train_params

def _get_config_path(project_root, dataset_name, model_type,stat_type):
    """
    根据规则构建配置文件路径。
    假设结构: configs/{dataset_name}/{model_name}/{model_name}_{stat_type}_{dataset_name}_fixed.json
    例如: configs/sgd/nmmr_q/nmmr_q_u_sgd_s1.json
    """
    model_name = f"nmmr_{model_type}" # nmmr_q 或 nmmr_h
    file_name = f"{model_name}_{stat_type}_{dataset_name}_fixed.json"
    
    config_path = project_root / "configs" / dataset_name / model_name / file_name
    
    
    return config_path

def run_double_robust_experiment(dataset_name: str, n_trials: int, rhc_mode: str = 'repeated'):
    """RHC path only (精简与config管理.md 之后新的 P0/E1/E2 SGD 路径见
    run_sgd_protocol_experiment() below -- SGD 和 RHC 的参数集合、职责差异太大，
    合并进同一个函数签名只会让两边都变得难读)。"""
    if dataset_name != 'rhc':
        raise ValueError(
            f"run_double_robust_experiment only serves dataset_name='rhc'; got {dataset_name!r}. "
            "SGD now goes through run_sgd_protocol_experiment()."
        )
    print(f"开始双重稳健性实验")
    print(f"数据集: {dataset_name}")
    print(f"Trials: {n_trials}")
    print(f"RHC 模式: {rhc_mode}")

    device = get_device()
    base_output_path = Path("predicts") / dataset_name / "dr_results"
    base_output_path.mkdir(parents=True, exist_ok=True)

    stat = ['u', 'v']
    _run_rhc_dr(device, base_output_path, stat, n_trials, dataset_name, rhc_mode)


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
    SGD 的 P0/E1/E2 入口 (plan/p0_e1_e2/01_shared_protocol.md section 9):
    只读取已通过 confirm 的正式 FrozenConfig，不再读取旧
    `configs/sgd/.../*_fixed.json`，不内联运行 Optuna。实际的场景、分折、
    NMMR 调度和结果写入全部在 src/experiments/sgd_experiment_protocol.py
    (本轮唯一的共用 protocol 实现)。

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


def _compute_pdr(dataset, q_preds, h_preds_dict):
    """计算 PDR 估计量"""
    h_fact = h_preds_dict['h_fact']
    h_1 = h_preds_dict['h_1']
    h_0 = h_preds_dict['h_0']

    Y_np = dataset.Y.cpu().numpy()
    A_np = dataset.A.cpu().numpy()

    weight = (2 * A_np - 1)
    residual = Y_np - h_fact
    term1 = weight * q_preds * residual
    term2 = h_1 - h_0

    pdr_individual = term1 + term2
    return np.mean(pdr_individual)


def _load_rhc_full_dataset(data_cfg, device):
    """根据单个 config 的 use_all_X 加载 RHC 全量数据"""
    rhc_data_dir = data_cfg.get('data_path', './data/right_heart_catheterization')
    use_all_x = data_cfg.get('use_all_X', False)
    train_ds = RHCDataset(split='train', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
    val_ds = RHCDataset(split='val', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
    test_ds = RHCDataset(split='test', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
    return MergedDataset([train_ds, val_ds, test_ds])


def _run_rhc_dr(device, base_output_path, stat, n_trials, dataset_name, rhc_mode='repeated'):
    """
    RHC 双重稳健实验，支持两种模式：
    - repeated: 在同一数据上用不同随机种子做 repeated cross-fitting，消除拆分噪声
    - bootstrap: 每次 trial 从全量数据有放回抽样，在 bootstrap 样本上 cross-fitting，
                 用于构造考虑数据采样不确定性的统计置信区间

    Q 和 H 分别根据各自 config 中的 use_all_X 加载数据集：
    - U-statistic: use_all_X=True  (49 features)
    - V-statistic: use_all_X=False (22 features)
    """
    for stat_type in stat:
        q_config_path = _get_config_path(project_root, dataset_name, "q", stat_type)
        h_config_path = _get_config_path(project_root, dataset_name, "h", stat_type)
        print(f"Loading Q Config: {q_config_path.name}")
        data_cfg_q, train_params_q = load_config_content(q_config_path, dataset_name)

        print(f"Loading H Config: {h_config_path.name}")
        data_cfg_h, train_params_h = load_config_content(h_config_path, dataset_name)

        # 分别从 Q 和 H 的 config 中读取 use_all_X
        use_all_x_q = data_cfg_q.get('use_all_X', False)
        use_all_x_h = data_cfg_h.get('use_all_X', False)

        # Q 和 H 的 use_all_X 相同时共用一份数据，不同时分别加载
        print(f"\n加载 RHC 数据集: Q(use_all_X={use_all_x_q}), H(use_all_X={use_all_x_h})")
        full_dataset_q = _load_rhc_full_dataset(data_cfg_q, device)
        if use_all_x_q == use_all_x_h:
            full_dataset_h = full_dataset_q
        else:
            full_dataset_h = _load_rhc_full_dataset(data_cfg_h, device)
        print(f"Q 数据样本数: {len(full_dataset_q)}, X维度: {full_dataset_q.X.shape[1]}")
        print(f"H 数据样本数: {len(full_dataset_h)}, X维度: {full_dataset_h.X.shape[1]}")

        # 从 config 读取各自的 cross-fitting 折数
        n_splits_q = int(train_params_q.get('n_splits', 5))
        n_splits_h = int(train_params_h.get('n_splits', 5))
        print(f"Cross-fitting 折数: Q={n_splits_q}, H={n_splits_h}")

        print(f"\n{'='*60}")
        print(f"Running RHC DR ({stat_type}-statistic, mode={rhc_mode})")
        print(f"{'='*60}")

        results_list = []

        for i in range(n_trials):
            # 种子只依赖 i(与 stat_type 无关)，保证同一个 trial i 在 U/V 两轮中
            # 使用相同的 bootstrap 重抽样/初始化规则。
            seeds = make_seed_bundle(1000, i)
            q_init_seed = make_seed_bundle(seeds.init_seed, 0).init_seed
            h_init_seed = make_seed_bundle(seeds.init_seed, 1).init_seed

            if rhc_mode == 'bootstrap':
                # 每次 trial 有放回抽样，产生新的 bootstrap 数据集
                dataset_q = BootstrapDataset(full_dataset_q, seed=seeds.bootstrap_seed)
                dataset_h = BootstrapDataset(full_dataset_h, seed=seeds.bootstrap_seed) if full_dataset_h is not full_dataset_q else dataset_q
            else:
                # repeated 模式：数据不变，仅通过 split_seed 改变 cross-fitting 折划分
                dataset_q = full_dataset_q
                dataset_h = full_dataset_h

            # --- Q ---
            q_res = NMMR_Q_experiment(
                dataset_name=dataset_name,
                data_configs=data_cfg_q,
                train_params=train_params_q,
                log_folder=None,
                random_seed=seeds.split_seed,
                dataset=dataset_q,
                init_seed=q_init_seed,
            )
            pipw_val = q_res['PIPW']
            q_preds = q_res['q_preds']

            # --- H ---
            h_res = NMMR_H_experiment(
                dataset_name=dataset_name,
                data_configs=data_cfg_h,
                train_params=train_params_h,
                log_folder=None,
                random_seed=seeds.split_seed,
                dataset=dataset_h,
                init_seed=h_init_seed,
            )
            por_val = h_res['POR']

            # --- PDR: 使用 Q 的数据集（包含 Y, A）和 Q/H 的预测 ---
            pdr_val = _compute_pdr(dataset_q, q_preds, h_res['h_preds'])

            results_list.append({
                "Trial": i,
                "PIPW": pipw_val,
                "POR": por_val,
                "PDR": pdr_val,
                "split_seed": seeds.split_seed,
                "init_seed": seeds.init_seed,
                "bootstrap_seed": seeds.bootstrap_seed if rhc_mode == 'bootstrap' else None,
            })

        df_res = pd.DataFrame(results_list)
        out_file = base_output_path / f"{stat_type}_rhc_{rhc_mode}.csv"
        df_res.to_csv(out_file, index=False)

        print(f"\n结果已保存: {out_file}")
        print(f"均值 -> PIPW: {df_res['PIPW'].mean():.4f}, POR: {df_res['POR'].mean():.4f}, PDR: {df_res['PDR'].mean():.4f}")
        print(f"标准差 -> PIPW: {df_res['PIPW'].std():.4f}, POR: {df_res['POR'].std():.4f}, PDR: {df_res['PDR'].std():.4f}")
        if rhc_mode == 'bootstrap' and n_trials >= 20:
            # 输出 95% bootstrap 置信区间
            for col in ['PIPW', 'POR', 'PDR']:
                lo = np.percentile(df_res[col], 2.5)
                hi = np.percentile(df_res[col], 97.5)
                print(f"  {col} 95% CI: [{lo:.4f}, {hi:.4f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='sgd',
        choices=['sgd', 'rhc'],
        help="数据集名称: 'sgd' 走新的 P0/E1/E2 共用 protocol, 'rhc' 走未改动的旧路径"
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=1,
        help="重复实验次数 (仅 RHC)"
    )
    parser.add_argument(
        '--rhc_mode',
        type=str,
        default='repeated',
        choices=['repeated', 'bootstrap'],
        help="RHC 实验模式: repeated (重复交叉拟合) 或 bootstrap (有放回抽样构造置信区间)"
    )
    # --- 以下仅用于 --dataset_name sgd (plan/p0_e1_e2/01_shared_protocol.md section 9) ---
    parser.add_argument('--run_stage', type=str, default='P0', help="P0/E1/E2 (仅 SGD)")
    parser.add_argument('--scenarios', type=str, default='S1', help="逗号分隔的场景 id, 如 'S1' (仅 SGD)")
    parser.add_argument(
        '--methods', type=str, default='nmmr_u,nmmr_v',
        help=f"逗号分隔的方法, 本轮只支持 {ALLOWED_METHODS} (仅 SGD)"
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

    if args.dataset_name == 'rhc':
        run_double_robust_experiment(
            dataset_name=args.dataset_name,
            n_trials=args.trials,
            rhc_mode=args.rhc_mode,
        )
    else:
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
