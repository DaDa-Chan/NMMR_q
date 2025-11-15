import torch
import numpy as np
import copy  
import optuna 
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.models.NMMRq.nmmr_q_trainers import NMMR_Q_Trainer_SGD
from src.data.ate.sgd_pv import generate_data
from src.data.ate.data_class import SGDDataset, RHCDataset


def _locate_split_file(
    base_path: Path,
    explicit_name: Optional[str],
    fallback_names: List[str],
) -> Optional[Path]:
    
    candidates: List[str] = []
    if explicit_name:
        candidates.append(explicit_name)
    candidates.extend(fallback_names)

    for name in candidates:
        candidate_path = base_path / name
        if candidate_path.exists():
            return candidate_path
    return None

def _parse_optuna_space(
    train_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    将 train_params 拆分为 固定的 (base) 和 可变的 (search_space)。
    """
    base_params = {}
    search_space = {}
    for key, value in train_params.items():
        if isinstance(value, dict) and "type" in value:
            # 这是一个 Optuna 搜索参数
            search_space[key] = value
        else:
            # 这是一个固定的基础参数
            base_params[key] = value
    return base_params, search_space

def _suggest_from_trial(
    trial: optuna.Trial, param_name: str, config: Dict[str, Any]
) -> Any:
    """
    根据 config 字典，从 Optuna trial 中获取一个建议值。
    """
    param_type = config["type"]
    if param_type == "categorical":
        return trial.suggest_categorical(param_name, config["choices"])
    elif param_type == "float":
        return trial.suggest_float(
            param_name,
            config["low"],
            config["high"],
            log=config.get("log", False),
        )
    elif param_type == "int":
        return trial.suggest_int(
            param_name,
            config["low"],
            config["high"],
            step=config.get("step", 1),
        )
    else:
        raise ValueError(f"未知的 Optuna 参数类型: {param_type}")

def _run_optuna_tuning(
    dataset,
    n_splits: int,
    n_trials: int,
    base_train_params: Dict[str, Any],
    search_space: Dict[str, Any],
    data_configs: Dict[str, Any],
    random_seed: int,
) -> Dict[str, Any]:
    """
    在给定数据集上执行 K-fold CV 以进行 Optuna 调参。
    返回最优的超参数组合。
    """
    if not search_space:
        print("[Optuna] 未提供搜索空间，跳过调优。")
        return base_train_params

    # 减少 Optuna 日志
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    batch_size = base_train_params["batch_size"]
    cv_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    def objective(trial: optuna.Trial) -> float:
        # 为当前试验生成参数
        current_params = copy.deepcopy(base_train_params)
        for param_name, config in search_space.items():
            current_params[param_name] = _suggest_from_trial(
                trial, param_name, config
            )

        fold_val_losses: List[float] = []

        # 运行 K-fold CV 来评估这组参数
        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_kfold.split(np.arange(len(dataset))), start=1
        ):
            # 调优时不在 fold 级别保存日志 (dump_folder=None)
            trainer = _create_trainer(
                data_configs, current_params, random_seed, dump_folder=None
            )

            train_loader = DataLoader(
                dataset=Subset(dataset, train_idx.tolist()),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                dataset=Subset(dataset, val_idx.tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            try:
                model = trainer.train(train_loader, val_loader)
                val_loss = _evaluate_q_loss(
                    trainer, model, dataset, val_idx.tolist()
                )
                fold_val_losses.append(val_loss)
            
            except Exception as e:
                print(f"[Optuna] Trial {trial.number} Fold {fold_idx} 失败: {e}")
                # 发生错误（例如 NaN 
                # loss）时，返回一个极差的值
                return float("inf")
            
            finally:
                # 清理内存
                del model, trainer, train_loader, val_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 返回 K-fold 的平均验证损失
        avg_val_loss = np.mean(fold_val_losses)
        return avg_val_loss

    # 运行 Optuna Study
    print(
        f"[Optuna] 开始 {n_trials} 次试验的 K-fold (k={n_splits}) 调优..."
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n[Optuna] 调优完成。")
    print(f"  最佳 Trial: {study.best_trial.number}")
    print(f"  最佳 q-loss: {study.best_value:.6f}")
    print(f"  最佳参数: {study.best_params}")

    # 合并并返回最终的最优参数
    final_best_params = {**base_train_params, **study.best_params}
    return final_best_params

def _run_cross_fitting(dataset,
                       n_splits: int,
                       train_params: Dict[str, Any],
                       data_configs: Dict[str, Any],
                       random_seed: int,
                       dump_folder: Optional[Path],
                       tag: str) -> Tuple[torch.Tensor, List[float]]:
    """
    对任意数据集执行 cross fitting:
    - 将数据划分为 n_splits 折
    - 在每折上训练模型（fold 内部再 8:2 划分 train/val）
    - 在其余折上直接计算 ATE_i，最后求均值
    """
    batch_size = train_params['batch_size']
    cf_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + 1024)

    fold_predictions: List[torch.Tensor] = []
    ate_values: List[float] = []

    for fold_idx, (eval_idx, train_idx) in enumerate(cf_kfold.split(np.arange(len(dataset))), start=1):
        print(f"\n[CF-{tag}] Fold {fold_idx}/{n_splits}")
        train_fold_indices = eval_idx.tolist()   # 当前折用于训练
        eval_fold_indices = train_idx.tolist()   # 剩余样本用于计算 ATE

        fold_folder = None
        if dump_folder is not None:
            fold_folder = dump_folder / f"fold_{fold_idx}"
            fold_folder.mkdir(parents=True, exist_ok=True)

        trainer = _create_trainer(data_configs, train_params, random_seed, fold_folder)

        train_loader, val_loader = _build_loaders_for_fold(
            dataset=dataset,
            indices=train_fold_indices,
            batch_size=batch_size,
            seed=random_seed + fold_idx,
        )
        model = trainer.train(train_loader, val_loader)

        eval_view = _build_dataset_view(dataset, eval_fold_indices)
        fold_ate = trainer.predict(model, eval_view)
        fold_predictions.append(fold_ate)
        ate_values.append(fold_ate.item())



        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not fold_predictions:
        raise RuntimeError("Cross-fitting 阶段未生成任何预测结果。")

    ate_tensor = torch.stack(fold_predictions).mean()
    print(f"[CF-{tag}] 平均 ATE: {ate_tensor.item():.6f}")
    return ate_tensor, ate_values

def _create_trainer(data_configs, train_params, random_seed, dump_folder):
    """
    目前 SGD 与 RHC 共用同一 Trainer，若未来需要可在此扩展。
    """
    return NMMR_Q_Trainer_SGD(
        data_configs=data_configs,
        train_params=train_params,
        random_seed=random_seed,
        dump_folder=dump_folder,
    )

def _build_loaders_for_fold(dataset,
                            indices: List[int],
                            batch_size: int,
                            seed: int):
    """
    将 fold 内部再划分为 train/val，以便 cross-fitting 阶段训练。
    """
    if len(indices) == 0:
        raise ValueError("fold 中没有样本。")

    rng = np.random.RandomState(seed)
    shuffled = np.array(indices)
    rng.shuffle(shuffled)
    split = max(1, int(0.8 * len(shuffled)))
    train_idx = shuffled[:split].tolist()
    val_idx = shuffled[split:].tolist() if split < len(shuffled) else shuffled[:split].tolist()

    train_loader = DataLoader(
        dataset=Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        dataset=Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader

def _build_dataset_view(dataset, indices: List[int]):
    """
    创建仅包含所选索引的“视图”，以便 trainer.predict 可以像操作完整数据集一样使用。
    """
    if len(indices) == 0:
        raise ValueError("评估折没有样本。")
    reference_tensor = getattr(dataset, 'X', getattr(dataset, 'A', None))
    base_device = reference_tensor.device if isinstance(reference_tensor, torch.Tensor) else torch.device('cpu')
    default_dtype = reference_tensor.dtype if isinstance(reference_tensor, torch.Tensor) else torch.float32
    index_tensor = torch.as_tensor(indices, device=base_device, dtype=torch.long)
    view = type('DatasetView', (), {})()
    dataset_len = len(dataset)

    for attr in ['X', 'A', 'Z', 'W', 'U', 'Y']:
        tensor = getattr(dataset, attr, None)
        if tensor is None or tensor.shape[0] != dataset_len:
            # 若属性不存在或长度不匹配（例如 RHC 没有 U），则用 0 填充
            tensor = torch.zeros(
                (dataset_len, 1),
                device=base_device,
                dtype=default_dtype,
            )
        setattr(view, attr, tensor.index_select(0, index_tensor))
    return view

def _evaluate_q_loss(trainer,
                     model: torch.nn.Module,
                     dataset,
                     indices: List[int]) -> float:
    """
    计算指定子集上的 q-loss（用于交叉验证评分）。
    """
    view = _build_dataset_view(dataset, indices)
    with torch.no_grad():
        A = view.A.to(trainer.device)
        W = view.W.to(trainer.device)
        X = view.X.to(trainer.device)
        Z = view.Z.to(trainer.device)
        
        inputs = torch.cat((A, Z, X), dim=1)
        preds = model(inputs)
        loss, _ = trainer.q_loss(q_a_hat=preds, a=A, w=W, x=X)
    return float(loss.item())


def NMMR_Q_experiment(dataset_name: str,
                      data_configs: Dict[str, Any],
                      train_params: Dict[str, Any],
                      dump_folder: Path,
                      random_seed: int):
    """
    运行 NMMR-Q 实验:
      * SGD: 先在 train 集上做 K-fold CV，再在 cf 集上 cross fitting。
      * RHC: 直接在指定 split 上执行 cross fitting（后续可扩展其它流程）。
    """
    print(f"--- 运行 NMMR-Q 实验 ---")
    print(f"数据集: {dataset_name}")
    print(f"随机种子: {random_seed}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    dataset_key = dataset_name.lower()
    train_params_copy = copy.deepcopy(train_params)
    base_params, search_space = _parse_optuna_space(train_params_copy)
    
    n_splits = int(base_params.get('n_splits', 5))
    hpo_n_trials = int(base_params.pop('hpo_n_trials', 20))
    
    final_train_params: Dict[str, Any] = {}


    if dataset_key == 'sgd':
        
        data_base_path = Path(data_configs['data_path'])
        
        n = 10000
        train_df = generate_data(n_samples=n)
        cf_df = generate_data(n_samples=n)

        output_dir = data_base_path
        output_dir.mkdir(parents=True, exist_ok=True)
        sgd_train = output_dir / 'sgd_train.csv'
        sgd_cf = output_dir / 'sgd_cf.csv'

        train_df.to_csv(sgd_train, index=False)
        cf_df.to_csv(sgd_cf, index=False)
        
        train_path = _locate_split_file(
            data_base_path,
            data_configs.get('train_file'),
            ['sgd_train.csv', 'train.csv'],
        )
        if train_path is None:
            raise FileNotFoundError(f"未在 {data_base_path} 找到 SGD 训练数据。")

        cf_path = _locate_split_file(
            data_base_path,
            data_configs.get('cf_file'),
            ['sgd_cf.csv', 'sgd_test.csv', 'test.csv'],
        )
        if cf_path is None:
            raise FileNotFoundError(f"未在 {data_base_path} 找到 cross-fitting 数据 (例如 sgd_cf.csv)。")

        base_dataset = SGDDataset(train_path, device=device)
        cf_dataset = SGDDataset(cf_path, device=device)

        # 在 SGD 训练集 (base_dataset) 上执行 HPO 调参
        
        if search_space:
            print("\n===> 开始 SGD 训练集 K-fold (Optuna 调参)")
            final_train_params = _run_optuna_tuning(
                dataset=base_dataset,
                n_splits=n_splits,
                n_trials=hpo_n_trials,
                base_train_params=base_params,
                search_space=search_space,
                data_configs=data_configs,
                random_seed=random_seed,
            )
        else:
            print("\n===> (跳过调参) 使用 config 中的固定参数")
            final_train_params = base_params

        print("\n===> 在 SGD CF 数据集上执行 cross fitting (使用最优/固定参数)")
        ate_estimate, ate_values = _run_cross_fitting(
            dataset=cf_dataset,
            n_splits=n_splits,
            train_params=final_train_params,
            data_configs=data_configs,
            random_seed=random_seed,
            dump_folder=dump_folder,
            tag="sgd_cf",
        )

    elif dataset_key == 'rhc':
        rhc_split = data_configs.get('cf_split', 'train')
        use_all_x = data_configs.get('use_all_X', False)
        rhc_data_dir = data_configs.get('data_path')
        cf_dataset = RHCDataset(
            split=rhc_split,
            use_all_x=use_all_x,
            data_dir=rhc_data_dir,
            device=device,
        )

        print("\n===> 在 RHC 数据集上执行 cross fitting (无 CV)")
        ate_estimate, ate_values = _run_cross_fitting(
            dataset=cf_dataset,
            n_splits=n_splits,
            train_params=train_params,
            data_configs=data_configs,
            random_seed=random_seed,
            dump_folder=None,
            tag="rhc_cf",
        )

    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    print(f"\n--- 最终 ATE (Seed {random_seed}) ---")
    print(f"ATE = {ate_estimate.item():.4f}")
    print("----------------------------------")





