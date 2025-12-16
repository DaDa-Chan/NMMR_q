import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

# --- 添加项目根目录到 Python 路径 ---
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ----------------------------------------

# 导入新的实验函数
from src.models.NMMRq.nmmr_q_experiments import NMMR_Q_experiment
from src.models.NMMRh.nmmr_h_experiments import NMMR_H_experiment
from src.models.LINEAR.linear_model import LINEAR_experiment


def load_configs(config_path_str: str, dataset_name: str) -> Tuple[Dict, Dict]:
    """
    加载并合并配置文件。
    它从 JSON 文件中提取特定数据集的数据配置和通用的训练参数。
    """
    config_path = Path(config_path_str)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    
    # 加载特定数据集的配置
    if dataset_name not in config_json['data_configs']:
        raise KeyError(f"在 {config_path} 中未找到数据集 '{dataset_name}' 的配置。")
    data_configs = config_json['data_configs'][dataset_name]

    # 加载训练参数
    if 'train_params' not in config_json:
        train_params = {}
    else:
        train_params = config_json['train_params']
    
    return data_configs, train_params

def set_random_seed(seed: int):
    """设置所有库的随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True,
        choices=['sgd', 'rhc'],
        help="要使用的数据集 (例如 'sgd' 或 'rhc')"
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        help="要运行的模型 (例如 'nmmr_q')"
    )
    parser.add_argument(
        '--config_path', 
        type=str, 
        required=True,
        help="指向 JSON 配置文件的路径"
    )
    parser.add_argument(
        '--n_seeds', 
        type=int, 
        default=1,
        help="要运行的随机种子的数量"
    )
    parser.add_argument(
        '--log_folder', 
        type=str, 
        default="logs",
        help="保存结果的根文件夹"
    )
    # n_jobs 在 DataLoader 太需要，但为保持兼容性保留
    parser.add_argument(
        '--predicts_path', 
        type=str, 
        default="predicts",
        help="预测文件地址"
    )
    
    args = parser.parse_args()

    # 1. 加载配置
    print(f"加载配置: {args.config_path}")
    data_configs, train_params = load_configs(args.config_path, args.dataset_name)

    # 2. 创建结果文件夹
    config_name = Path(args.config_path).stem  # 例如 'nmmr_q_sgd_config'
    experiment_log_folder = Path(args.log_folder) / args.dataset_name / args.model_name / config_name
    experiment_predicts_folder = Path(args.predicts_path) / args.dataset_name / args.model_name

    # 3. 循环运行多个随机种子
    for i in range(args.n_seeds):
        random_seed = i
        print(f"\n--- 开始运行 Seed {random_seed + 1} / {args.n_seeds} ---")
        
        # 为当前种子创建子文件夹
        seed_log_folder = experiment_log_folder / str(random_seed)
        seed_log_folder.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        set_random_seed(random_seed)
        
        # 根据模型名称调用相应的实验函数
        if args.model_name == 'nmmr_q':
            NMMR_Q_experiment(
                dataset_name=args.dataset_name,
                data_configs=data_configs,
                train_params=train_params,
                log_folder=seed_log_folder,
                random_seed=random_seed
            )
        elif args.model_name == 'nmmr_h':  
            NMMR_H_experiment(
                dataset_name=args.dataset_name,
                data_configs=data_configs,
                train_params=train_params,
                log_folder=seed_log_folder,
                random_seed=random_seed
            )
        elif args.model_name == 'linear':
            LINEAR_experiment(
                data_configs=data_configs
            )       
        else:
            print(f"Unknown model: {args.model_name}")
            
    print("\n所有实验运行完毕。")

if __name__ == '__main__':
    main()
