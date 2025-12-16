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
from src.data.ate.data_class import SGDDataset
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

def run_double_robust_experiment(dataset_name: str, n_trials: int, n_samples: int = 2000):
    print(f"开始双重稳健性实验")
    print(f"数据集: {dataset_name}")
    print(f"Trials: {n_trials}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_output_path = Path("predicts") / dataset_name / "dr_results"
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    scenarios = [1, 2, 3, 4]
    stat = ['u', 'v']
    
    
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
                seed = i + 1000 # 避免与之前的实验种子重叠
                # print(f"\n[Scenario {scenario}] Trial {i+1}/{n_trials} (Seed {seed})")

                # 2. 生成统一数据
                # 注意: n_samples 可以从命令行传，也可以读取 config，这里优先使用命令行/函数参数
                df = generate_data(n_samples=n_samples, scenario=scenario)
            
                # 封装为 Dataset
                # 注意: 这里的 dataset_name 仅用于类选择逻辑，具体数据在 df 里
                if dataset_name == 'sgd':
                    dataset = SGDDataset(csv_path='', df=df, device=device)
                else:
                    raise NotImplementedError("目前仅支持 SGD 数据的生成逻辑")

                # 3. 运行 NMMR-Q (PIPW)
                # 传入 dataset, 使用 train_params_q
                q_res = NMMR_Q_experiment(
                    dataset_name=dataset_name,
                    data_configs=data_cfg_q,
                    train_params=train_params_q,
                    log_folder=None,
                    random_seed=seed,
                    dataset=dataset 
                )
                pipw_val = q_res['PIPW']
                q_preds = q_res['q_preds'] # shape (N, 1)

                # 4. 运行 NMMR-H (POR)
                # 传入相同的 dataset, 使用 train_params_h
                h_res = NMMR_H_experiment(
                    dataset_name=dataset_name,
                    data_configs=data_cfg_h,
                    train_params=train_params_h,
                    log_folder=None,
                    random_seed=seed,
                    dataset=dataset
                )
                por_val = h_res['POR']
                h_preds_dict = h_res['h_preds']
            
                h_fact = h_preds_dict['h_fact'] # (N, 1)
                h_1 = h_preds_dict['h_1']       # (N, 1)
                h_0 = h_preds_dict['h_0']       # (N, 1)
            
                # 5. 计算 PDR (Proximal Doubly Robust)
                # Formula: (-1)^(1-A) * q * (Y - h_fact) + (h_1 - h_0)
            
                Y_np = dataset.Y.cpu().numpy()
                A_np = dataset.A.cpu().numpy()

                weight = (2 * A_np - 1) 
                residual = Y_np - h_fact
                term1 = weight * q_preds * residual
            
                # Term 2: Outcome contrast
                term2 = h_1 - h_0
            
                pdr_individual = term1 + term2
                pdr_val = np.mean(pdr_individual)
            
                # print(f"  -> PIPW: {pipw_val:.4f} | POR: {por_val:.4f} | PDR: {pdr_val:.4f}")
            
                results_list.append({
                    "Trial": i,
                    "PIPW": pipw_val,
                    "POR": por_val,
                    "PDR": pdr_val
                })
        
            # 保存结果
            df_res = pd.DataFrame(results_list)
            out_file = base_output_path / f"{loss_name}_s{scenario}.csv"
            df_res.to_csv(out_file, index=False)
        
            print(f"结果已保存: {out_file}")
            print(f"均值 -> PIPW: {df_res['PIPW'].mean():.4f}, POR: {df_res['POR'].mean():.4f}, PDR: {df_res['PDR'].mean():.4f}")

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
        help="生成数据的样本数量"
    )
    
    args = parser.parse_args()
    
    run_double_robust_experiment(
        dataset_name=args.dataset_name, 
        n_trials=args.trials,
        n_samples=args.n_samples
    )
    