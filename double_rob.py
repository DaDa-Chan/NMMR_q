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

from src.data.ate.sgd_pv import generate_data
from src.data.ate.data_class import SGDDataset, RHCDataset, MergedDataset, BootstrapDataset
from src.models.NMMRq.nmmr_q_experiments import NMMR_Q_experiment
from src.models.NMMRh.nmmr_h_experiments import NMMR_H_experiment

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

def run_double_robust_experiment(dataset_name: str, n_trials: int, n_samples: int = 2000,
                                  rhc_mode: str = 'repeated'):
    print(f"开始双重稳健性实验")
    print(f"数据集: {dataset_name}")
    print(f"Trials: {n_trials}")
    if dataset_name == 'rhc':
        print(f"RHC 模式: {rhc_mode}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_output_path = Path("predicts") / dataset_name / "dr_results"
    base_output_path.mkdir(parents=True, exist_ok=True)

    stat = ['u', 'v']

    if dataset_name == 'sgd':
        _run_sgd_dr(device, base_output_path, stat, n_trials, n_samples, dataset_name)
    elif dataset_name == 'rhc':
        _run_rhc_dr(device, base_output_path, stat, n_trials, dataset_name, rhc_mode)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")


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


def _run_single_trial(dataset_name, dataset, data_cfg_q, train_params_q, data_cfg_h, train_params_h, seed):
    """对一个 dataset 运行一次 Q + H + PDR"""
    q_res = NMMR_Q_experiment(
        dataset_name=dataset_name,
        data_configs=data_cfg_q,
        train_params=train_params_q,
        log_folder=None,
        random_seed=seed,
        dataset=dataset
    )
    pipw_val = q_res['PIPW']
    q_preds = q_res['q_preds']

    h_res = NMMR_H_experiment(
        dataset_name=dataset_name,
        data_configs=data_cfg_h,
        train_params=train_params_h,
        log_folder=None,
        random_seed=seed,
        dataset=dataset
    )
    por_val = h_res['POR']
    pdr_val = _compute_pdr(dataset, q_preds, h_res['h_preds'])

    return pipw_val, por_val, pdr_val


def _run_sgd_dr(device, base_output_path, stat, n_trials, n_samples, dataset_name):
    """SGD 双重稳健实验：遍历 scenario x stat_type，每次重新生成数据"""
    scenarios = [1, 2, 3, 4]

    for stat_type in stat:
        q_config_path = _get_config_path(project_root, dataset_name, "q", stat_type)
        h_config_path = _get_config_path(project_root, dataset_name, "h", stat_type)
        print(f"Loading Q Config: {q_config_path.name}")
        data_cfg_q, train_params_q = load_config_content(q_config_path, dataset_name)

        print(f"Loading H Config: {h_config_path.name}")
        data_cfg_h, train_params_h = load_config_content(h_config_path, dataset_name)

        loss_name = stat_type
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Running Scenario {scenario}")
            print(f"{'='*60}")

            results_list = []

            for i in range(n_trials):
                seed = i + 1000
                df = generate_data(n_samples=n_samples, scenario=scenario)
                dataset = SGDDataset(csv_path='', df=df, device=device)

                pipw_val, por_val, pdr_val = _run_single_trial(
                    dataset_name, dataset,
                    data_cfg_q, train_params_q,
                    data_cfg_h, train_params_h,
                    seed
                )

                results_list.append({
                    "Trial": i,
                    "PIPW": pipw_val,
                    "POR": por_val,
                    "PDR": pdr_val
                })

            df_res = pd.DataFrame(results_list)
            out_file = base_output_path / f"{loss_name}_s{scenario}.csv"
            df_res.to_csv(out_file, index=False)

            print(f"结果已保存: {out_file}")
            print(f"均值 -> PIPW: {df_res['PIPW'].mean():.4f}, POR: {df_res['POR'].mean():.4f}, PDR: {df_res['PDR'].mean():.4f}")


def _run_rhc_dr(device, base_output_path, stat, n_trials, dataset_name, rhc_mode='repeated'):
    """
    RHC 双重稳健实验，支持两种模式：
    - repeated: 在同一数据上用不同随机种子做 repeated cross-fitting，消除拆分噪声
    - bootstrap: 每次 trial 从全量数据有放回抽样，在 bootstrap 样本上 cross-fitting，
                 用于构造考虑数据采样不确定性的统计置信区间
    """
    for stat_type in stat:
        q_config_path = _get_config_path(project_root, dataset_name, "q", stat_type)
        h_config_path = _get_config_path(project_root, dataset_name, "h", stat_type)
        print(f"Loading Q Config: {q_config_path.name}")
        data_cfg_q, train_params_q = load_config_content(q_config_path, dataset_name)

        print(f"Loading H Config: {h_config_path.name}")
        data_cfg_h, train_params_h = load_config_content(h_config_path, dataset_name)

        # 加载 RHC 全量数据（只需加载一次）
        rhc_data_dir = data_cfg_q.get('data_path', './data/right_heart_catheterization')
        use_all_x = data_cfg_q.get('use_all_X', False)
        print(f"\n加载 RHC 数据集 (use_all_X={use_all_x})...")
        train_ds = RHCDataset(split='train', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
        val_ds = RHCDataset(split='val', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
        test_ds = RHCDataset(split='test', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
        full_dataset = MergedDataset([train_ds, val_ds, test_ds])
        print(f"全量数据样本数: {len(full_dataset)}")

        print(f"\n{'='*60}")
        print(f"Running RHC DR ({stat_type}-statistic, mode={rhc_mode})")
        print(f"{'='*60}")

        results_list = []

        for i in range(n_trials):
            seed = i + 1000

            if rhc_mode == 'bootstrap':
                # 每次 trial 有放回抽样，产生新的 bootstrap 数据集
                dataset = BootstrapDataset(full_dataset, seed=seed)
            else:
                # repeated 模式：数据不变，仅通过 seed 改变 cross-fitting 折划分
                dataset = full_dataset

            pipw_val, por_val, pdr_val = _run_single_trial(
                dataset_name, dataset,
                data_cfg_q, train_params_q,
                data_cfg_h, train_params_h,
                seed
            )

            results_list.append({
                "Trial": i,
                "PIPW": pipw_val,
                "POR": por_val,
                "PDR": pdr_val
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
        help="数据集名称，用于寻找配置文件路径 (e.g. sgd)"
    )
    parser.add_argument(
        '--trials', 
        type=int, 
        default=1,
        help="重复实验次数"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=2000,
        help="生成数据的样本数量 (仅 SGD)"
    )
    parser.add_argument(
        '--rhc_mode',
        type=str,
        default='repeated',
        choices=['repeated', 'bootstrap'],
        help="RHC 实验模式: repeated (重复交叉拟合) 或 bootstrap (有放回抽样构造置信区间)"
    )

    args = parser.parse_args()

    run_double_robust_experiment(
        dataset_name=args.dataset_name,
        n_trials=args.trials,
        n_samples=args.n_samples,
        rhc_mode=args.rhc_mode
    )
    