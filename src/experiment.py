import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

# --- 添加项目根目录到 Python 路径 ---
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ----------------------------------------

from src.models.NMMRq.nmmr_q_experiments import NMMR_Q_experiment, train_q_standalone
from src.models.NMMRh.nmmr_h_experiments import NMMR_H_experiment, train_h_standalone
from src.models.LINEAR.linear_model import LINEAR_experiment
from src.models.experiment_common import _parse_optuna_space
from src.models import hpo_protocol as hp
from src.utils import set_random_seed, make_seed_bundle


def load_configs(config_path_str: str, dataset_name: str) -> Tuple[Dict, Dict]:
    """
    加载并合并配置文件。
    它从 JSON 文件中提取特定数据集的数据配置和通用的训练参数。

    This entrypoint only runs fixed-parameter training/evaluation (pilot HPO
    fix request item 8): hyperparameter search happens exclusively in
    scripts/run_pilot_protocol.py and ends in a frozen config. If the given
    config still has an Optuna search space (e.g. the un-tuned
    configs/sgd/nmmr_q/nmmr_q_u_sgd.json rather than a frozen config), this
    raises immediately instead of silently launching HPO inline.
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

    _, search_space = _parse_optuna_space(train_params)
    if search_space:
        raise ValueError(
            f"{config_path} still has an Optuna search space for {list(search_space)}. "
            "Run scripts/run_pilot_protocol.py first and provide a frozen config."
        )

    return data_configs, train_params


def load_data_configs_only(config_path_str: str, dataset_name: str) -> Dict:
    """
    For mode='traditional' (精简与config管理.md Phase D): --config_path now
    only supplies DGP/data settings (sample size, data seed, scenario) --
    train_params for this mode comes from --frozen_config_path instead, via
    hpo_protocol.FrozenConfig. Unlike load_configs(), this intentionally
    does not look at or validate the file's train_params at all (it may
    still carry an Optuna search space meant for scripts/run_pilot_protocol.py;
    that's fine here since this mode never reads it), so the same existing
    configs/sgd/.../*.json files can keep serving as the data source without
    needing a separate train-params-free copy.
    """
    config_path = Path(config_path_str)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r') as f:
        config_json = json.load(f)

    if dataset_name not in config_json['data_configs']:
        raise KeyError(f"在 {config_path} 中未找到数据集 '{dataset_name}' 的配置。")
    return config_json['data_configs'][dataset_name]


def load_frozen_config_for_standalone(frozen_config_path_str: str, model_name: str) -> Dict:
    """
    For mode='traditional': load and validate a user-selected, already
    pilot-confirmed HPO FrozenConfig (精简与config管理.md Phase D). Parses
    exclusively via hpo_protocol.FrozenConfig -- never re-implements its
    field layout here -- and returns the resolved train_params via
    to_train_params(). Checks (and raises a clear error on failure):

    1. confirm_status == "passed";
    2. the frozen config's method_id matches `model_name` (Q vs H) and is
       internally self-consistent (method_id's U/V suffix matches its own
       training_objective field);
    3. no leftover Optuna search space / hpo_n_trials key.

    Never auto-selects or copies anything from candidates/, never runs
    Optuna, never writes back to the frozen config file.
    """
    frozen_config_path = Path(frozen_config_path_str)
    if not frozen_config_path.exists():
        raise FileNotFoundError(f"Frozen config 未找到: {frozen_config_path}")

    frozen = hp.FrozenConfig.load(frozen_config_path)
    hp.validate_method_ids([frozen.method_id])

    model_kind, training_objective = frozen.method_id.split("-")
    expected_kind = {"nmmr_q": "Q", "nmmr_h": "H"}.get(model_name)
    if expected_kind is None:
        raise ValueError(
            f"mode='traditional' 目前只支持 model_name in ('nmmr_q', 'nmmr_h')，"
            f"收到 {model_name!r}。"
        )
    if model_kind != expected_kind:
        raise ValueError(
            f"{frozen_config_path} 的 method_id={frozen.method_id!r} 与 "
            f"--model_name={model_name!r} 不一致 (期望 {expected_kind}-*)。"
        )
    if training_objective != frozen.training_objective:
        raise ValueError(
            f"{frozen_config_path} 内部不一致: method_id={frozen.method_id!r} "
            f"暗示 training_objective={training_objective!r}，但其 training_objective "
            f"字段是 {frozen.training_objective!r}。"
        )
    if frozen.confirm_status != "passed":
        raise ValueError(
            f"{frozen_config_path} (method_id={frozen.method_id}) 的 confirm_status="
            f"{frozen.confirm_status!r}；只有 confirm_status=='passed' 的 frozen config "
            "才能用于 standalone/traditional 对比。"
        )

    train_params = frozen.to_train_params(model_kind)
    hp.validate_frozen_config(train_params)
    return train_params


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
        help="指向数据/DGP配置的 JSON 文件路径 (样本量、数据 seed、scenario 等)。"
             "mode='traditional' 时忽略其中的 train_params，固定训练参数改由 "
             "--frozen_config_path 提供。"
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
    parser.add_argument(
        '--predicts_path', 
        type=str, 
        default="predicts",
        help="预测文件地址"
    )   
    parser.add_argument(
        '--mode',
        type=str,
        default='cross_fitting',
        choices=['cross_fitting', 'traditional'],
        help="运行的模式: 'cross_fitting' (K-fold cross-fitting) or "
             "'traditional' (standalone: 用 --frozen_config_path 指定的正式 frozen "
             "config，在单个 train/test 划分上训练一个模型，作为非 cross-fitting 的横向对比)"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='saved_models',
        help="mode='traditional' 时 standalone 模型权重的输出目录 "
             "(仅在传入 --save_model 时才会写入该目录；不是预训练模型的输入目录)"
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help="mode='traditional' 时是否保存模型权重 (默认不保存，Monte Carlo 重复时避免堆积权重文件)"
    )
    parser.add_argument(
        '--frozen_config_path',
        type=str,
        default=None,
        help="mode='traditional' 时必填：指向用户已人工确认并放入 configs/sgd_frozen/ 的 "
             "正式 FrozenConfig JSON 文件 (例如 configs/sgd_frozen/q_u_frozen_config.json)"
    )

    args = parser.parse_args()

    # 1. 加载配置
    print(f"加载配置: {args.config_path}")
    if args.mode == 'traditional':
        if not args.frozen_config_path:
            raise ValueError(
                "mode='traditional' 需要 --frozen_config_path 指向已人工确认、"
                "confirm_status=='passed' 的正式 frozen config (configs/sgd_frozen/ 下的文件)。"
            )
        data_configs = load_data_configs_only(args.config_path, args.dataset_name)
        train_params = load_frozen_config_for_standalone(args.frozen_config_path, args.model_name)
        print(f"加载 frozen config: {args.frozen_config_path}")
    else:
        data_configs, train_params = load_configs(args.config_path, args.dataset_name)

    # 2. 创建结果文件夹
    config_name = Path(args.config_path).stem  # 例如 'nmmr_q_sgd_config'
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log_folder = Path(args.log_folder) / args.dataset_name / args.model_name / config_name / suffix
    experiment_predicts_folder = Path(args.predicts_path) / args.dataset_name / args.model_name

    # 3. 循环运行多个随机种子
    for i in range(args.n_seeds):
        run_seed = i + 42
        seeds = make_seed_bundle(run_seed)
        print(f"\n--- 开始运行 Seed {run_seed} (data={seeds.data_seed}, split={seeds.split_seed}, "
              f"init={seeds.init_seed}, tuning={seeds.tuning_seed}) ---")

        # 为当前种子创建子文件夹
        seed_log_folder = experiment_log_folder / str(run_seed)
        seed_log_folder.mkdir(parents=True, exist_ok=True)

        # 设置 init_seed 对应的全局随机状态
        set_random_seed(seeds.init_seed)

        if args.mode == 'traditional':
            print("模式: standalone (非 cross-fitting 单模型对比)")
            if args.model_name == 'nmmr_q':
                result = train_q_standalone(
                    data_configs,
                    train_params,
                    dataset_name=args.dataset_name,
                    scenario=data_configs.get('scenario', 1),
                    save_dir=args.model_dir,
                    log_folder=seed_log_folder,
                    random_seed=seeds.split_seed,
                    data_seed=seeds.data_seed,
                    init_seed=seeds.init_seed,
                    save_model=args.save_model,
                )
                print(f"[Standalone] 结果: {result}")
            elif args.model_name == 'nmmr_h':
                result = train_h_standalone(
                    data_configs,
                    train_params,
                    dataset_name=args.dataset_name,
                    scenario=data_configs.get('scenario', 1),
                    save_dir=args.model_dir,
                    log_folder=seed_log_folder,
                    random_seed=seeds.split_seed,
                    data_seed=seeds.data_seed,
                    init_seed=seeds.init_seed,
                    save_model=args.save_model,
                )
                print(f"[Standalone] 结果: {result}")

        else:
            if args.model_name == 'nmmr_q':
                NMMR_Q_experiment(
                    dataset_name=args.dataset_name,
                    data_configs=data_configs,
                    train_params=train_params,
                    log_folder=seed_log_folder,
                    random_seed=seeds.split_seed,
                    data_seed=seeds.data_seed,
                    init_seed=seeds.init_seed,
                    tuning_seed=seeds.tuning_seed,
                )
            elif args.model_name == 'nmmr_h':
                NMMR_H_experiment(
                    dataset_name=args.dataset_name,
                    data_configs=data_configs,
                    train_params=train_params,
                    log_folder=seed_log_folder,
                    random_seed=seeds.split_seed,
                    data_seed=seeds.data_seed,
                    init_seed=seeds.init_seed,
                    tuning_seed=seeds.tuning_seed,
                )
            elif args.model_name == 'linear':
                LINEAR_experiment(
                    data_configs=data_configs,
                    data_seed=seeds.data_seed,
                )
            else:
                print(f"Unknown model: {args.model_name}")

            print("\n所有实验运行完毕。")

if __name__ == '__main__':
    main()
