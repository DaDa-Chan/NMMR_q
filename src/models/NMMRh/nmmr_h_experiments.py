import torch
import numpy as np
import pandas as pd
import copy  
import optuna 
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


from src.models.NMMRh.nmmr_h_trainers import NMMR_H_Trainer, NMMR_H_Trainer_SGD, NMMR_H_Trainer_RHC, MLP_for_NMMR 
from src.models.NMMRh.nmmr_h_model import MLP_for_NMMR
from src.data.ate.sgd_pv import generate_data
from src.data.ate.data_class import SGDDataset, RHCDataset, MergedDataset


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
            search_space[key] = value
        else:
            base_params[key] = value
    return base_params, search_space

def _suggest_from_trial(
    trial: optuna.Trial, param_name: str, config: Dict[str, Any]
) -> Any:
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

def _create_trainer(data_configs, train_params, random_seed, dump_folder, data_name):
    """
    根据配置创建对应的 H Trainer
    """
    if data_name == 'sgd':
        return NMMR_H_Trainer_SGD(
            data_configs=data_configs,
            train_params=train_params,
            random_seed=random_seed,
            dump_folder=dump_folder,
        )
    elif data_name == 'rhc' :
        return NMMR_H_Trainer_RHC(
            data_configs=data_configs,
            train_params=train_params,
            random_seed=random_seed,
            dump_folder=dump_folder,
        )

def _build_loaders_for_fold(dataset, indices: List[int], batch_size: int, seed: int):
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
    构建数据视图，用于评估和预测
    修复版本：移除了严格的长度检查，防止因 dataset 长度与 tensor 维度元数据不一致导致的静默零填充问题。
    """
    if len(indices) == 0:
        raise ValueError("评估折没有样本。")

    reference_tensor = getattr(dataset, 'X', getattr(dataset, 'A', None))
    base_device = reference_tensor.device if isinstance(reference_tensor, torch.Tensor) else torch.device('cpu')
    default_dtype = reference_tensor.dtype if isinstance(reference_tensor, torch.Tensor) else torch.float32
    index_tensor = torch.as_tensor(indices, device=base_device, dtype=torch.long)
    view = type('DatasetView', (), {})()

    target_len = len(indices)

    for attr in ['X', 'A', 'Z', 'W', 'U', 'Y']:
        tensor = getattr(dataset, attr, None)

        if tensor is not None:
            try:
                setattr(view, attr, tensor.index_select(0, index_tensor))
            except RuntimeError:
                setattr(view, attr, torch.zeros((target_len, 1), device=base_device, dtype=default_dtype))
        else:

            setattr(view, attr, torch.zeros((target_len, 1), device=base_device, dtype=default_dtype))
            
    return view

def _evaluate_h_loss(trainer, model: torch.nn.Module, dataset, indices: List[int]) -> float:
    """
    计算验证集上的 H-Loss (用于 HPO)
    """
    view = _build_dataset_view(dataset, indices)
    with torch.no_grad():
        A = view.A.to(trainer.device)
        W = view.W.to(trainer.device)
        Z = view.Z.to(trainer.device)
        X = view.X.to(trainer.device)
        Y = view.Y.to(trainer.device)

        k_matrix = trainer._compute_kernel_matrix(A, Z, X)

        model_input = torch.cat((A, W, X), dim=1)
        pred = model(model_input)

        loss = trainer.h_loss(pred, Y, k_matrix)
            
    return float(loss.item())

def _run_optuna_tuning(
    data_name:str,
    dataset,
    n_splits: int,
    n_trials: int,
    base_train_params: Dict[str, Any],
    search_space: Dict[str, Any],
    data_configs: Dict[str, Any],
    random_seed: int,
    log_folder: Optional[Path] = None,
    save_dir: Optional[Path] = None,   
    scenario: Optional[int] = 1    
) -> Dict[str, Any]:
    """
    1.在给定数据集上执行 K-fold CV 以进行 Optuna 调参。返回最优的超参数组合。
    2.如果提供了 save_dir, 则使用最佳参数在全量 dataset 上重新训练并保存模型。
    """
    
    if not search_space:
        print("[Optuna] 未提供搜索空间，跳过调优。")
        return base_train_params

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    batch_size = base_train_params["batch_size"]
    cv_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    def objective(trial: optuna.Trial) -> float:
        current_params = copy.deepcopy(base_train_params)
        for param_name, config in search_space.items():
            current_params[param_name] = _suggest_from_trial(
                trial, param_name, config
            )

        fold_val_losses: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_kfold.split(np.arange(len(dataset))), start=1
            ):
            
            trainer = _create_trainer(
                data_configs, current_params, random_seed, dump_folder=None, data_name=data_name
            )
            
            # 这里复用 _build_loaders_for_fold 也可以，或者直接构造
            t_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
            v_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)

            try:
                model = trainer.train(t_loader, v_loader)
                val_loss = _evaluate_h_loss(trainer, model, dataset, val_idx.tolist())
                fold_val_losses.append(val_loss)
            except Exception as e:
                print(f"[Optuna] Trial {trial.number} Fold {fold_idx} 失败: {e}")
                return float("inf")
            finally:
                if 'model' in locals(): del model
                if 'trainer' in locals(): del trainer
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        return np.mean(fold_val_losses)

    print(f"[Optuna] 开始 {n_trials} 次试验的 K-fold (k={n_splits}) 调优...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n[Optuna] 调优完成。")
    print(f"  最佳 Trial: {study.best_trial.number}")
    print(f"  最佳 h-loss: {study.best_value:.6f}")
    print(f"  最佳参数: {study.best_params}")
    
    final_best_params = {**base_train_params, **study.best_params}
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("\n[Refit] 使用最佳参数在全量数据上训练最终模型...")
        
        batch_size = final_best_params['batch_size']
        full_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        full_val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 

        refit_dump_folder = log_folder if log_folder else None

        final_trainer = _create_trainer(
            data_configs=data_configs,
            train_params=final_best_params, # 使用最佳参数
            random_seed=random_seed,
            dump_folder=refit_dump_folder, 
            data_name=data_name
        )

        final_model = final_trainer.train(full_train_loader, full_val_loader, verbose=1)

        stat_type = "u" if str(final_best_params.get('use_u_statistic', 'false')).lower() == 'true' else "v"
        model_filename = f"nmmr_h_{data_name}_{stat_type}_s{scenario}.pth"     
        torch.save(final_model.state_dict(), save_path / model_filename)
                   
        print(f"[Save] 模型已保存: {save_path / model_filename}")

    return final_best_params

def _run_cross_fitting(data_name:str,
                       dataset,
                       n_splits: int,
                       train_params: Dict[str, Any],
                       data_configs: Dict[str, Any],
                       random_seed: int,
                       log_folder: Optional[Path],
                       tag: str)  -> Tuple[torch.Tensor, List[float], Dict]:
    
    batch_size = train_params['batch_size']
    cf_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + 1024)
    
    fold_predictions: List[torch.Tensor] = []
    ate_values: List[float] = []
    
    indices = np.arange(len(dataset))
    dataset_len = len(dataset)

    h_fact_full = np.zeros((dataset_len, 1), dtype=np.float32)
    h_1_full = np.zeros((dataset_len, 1), dtype=np.float32)
    h_0_full = np.zeros((dataset_len, 1), dtype=np.float32)
    
    for fold_idx, (train_idx_np, eval_idx_np) in enumerate(cf_kfold.split(indices), start=1):
        print(f"\n[CF-{tag}] Fold {fold_idx}/{n_splits}")
        train_fold_indices = train_idx_np.tolist()
        eval_fold_indices = eval_idx_np.tolist()

        fold_folder = None
        if log_folder is not None:
            fold_folder = log_folder / f"fold_{fold_idx}"
            fold_folder.mkdir(parents=True, exist_ok=True)

        trainer = _create_trainer(data_configs, train_params, random_seed, fold_folder, data_name)
        
        train_loader, val_loader = _build_loaders_for_fold(
            dataset=dataset,
            indices=train_fold_indices,
            batch_size=batch_size,
            seed=random_seed + fold_idx,
        )
        
        # 训练
        model = trainer.train(train_loader, val_loader)

        # 预测 ATE
        eval_view = _build_dataset_view(dataset, eval_fold_indices)
        fold_ate = trainer.predict(model, eval_view)
        fold_predictions.append(fold_ate)
        ate_values.append(fold_ate.item())
        print(f"  Fold {fold_idx} ATE: {fold_ate.item():.4f}")
        
        with torch.no_grad():
            W = eval_view.W.to(trainer.device)
            X = eval_view.X.to(trainer.device)
            A = eval_view.A.to(trainer.device)
            
            # Factual h(A, W, X)
            inputs_fact = torch.cat((A, W, X), dim=1)
            h_fact = model(inputs_fact)
            
            # Counterfactual h(1, W, X)
            inputs_1 = torch.cat((torch.ones_like(A), W, X), dim=1)
            h_1 = model(inputs_1)
            
            # Counterfactual h(0, W, X)
            inputs_0 = torch.cat((torch.zeros_like(A), W, X), dim=1)
            h_0 = model(inputs_0)
            
            h_fact_full[eval_fold_indices] = h_fact.cpu().numpy()
            h_1_full[eval_fold_indices] = h_1.cpu().numpy()
            h_0_full[eval_fold_indices] = h_0.cpu().numpy()

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not fold_predictions:
        raise RuntimeError("Cross-fitting 阶段未生成任何预测结果。")

    ate_tensor = torch.stack(fold_predictions).mean()
    print(f"[CF-{tag}] 平均 ATE(POR): {ate_tensor.item():.6f}")
    return ate_tensor, ate_values, {
        "h_fact": h_fact_full,
        "h_1": h_1_full,
        "h_0": h_0_full
    }

def _run_traditional_prediction(
    dataset,
    train_params: Dict[str, Any],
    model_path: Path,
    device: str
) -> Tuple[torch.Tensor, Dict]:
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    
    print(f"[Traditional] Loading model from {model_path}")
    

    W_dim = dataset.W.shape[1]
    X_dim = dataset.X.shape[1]
    input_size = 1 + W_dim + X_dim  
    
    model = MLP_for_NMMR(input_dim = input_size, train_params=train_params)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    full_view = _build_dataset_view(dataset, list(range(len(dataset))))
    
    ate_val = NMMR_H_Trainer.predict(model, full_view)
    
    with torch.no_grad():
            W = full_view.W.to(device)
            X = full_view.X.to(device)
            A = full_view.A.to(device)
            
            # Factual h(A, W, X)
            inputs_fact = torch.cat((A, W, X), dim=1)
            h_fact = model(inputs_fact)
            
            # Counterfactual h(1, W, X)
            inputs_1 = torch.cat((torch.ones_like(A), W, X), dim=1)
            h_1 = model(inputs_1)
            
            # Counterfactual h(0, W, X)
            inputs_0 = torch.cat((torch.zeros_like(A), W, X), dim=1)
            h_0 = model(inputs_0)
            
            h_fact_full = h_fact.cpu().numpy()
            h_1_full = h_1.cpu().numpy()
            h_0_full = h_0.cpu().numpy()
            
    return ate_val, {
        "h_fact": h_fact_full,
        "h_1": h_1_full,
        "h_0": h_0_full
    }
# -------------------------------------------------------
# 主入口函数: NMMR_H_experiment
# -------------------------------------------------------

def NMMR_H_experiment(dataset_name: str,
                      data_configs: Dict[str, Any],
                      train_params: Dict[str, Any],
                      log_folder: Path,
                      random_seed: int,
                      dataset = None,
                      mode: str = 'cross_fitting',
                      model_dir: Optional[Path] = None
                      ):
    """
    运行 NMMR-H 实验 (求解 h 函数并计算 ATE)
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_key = dataset_name.lower()
    
    if mode == 'traditional':
        if model_dir is None:
            raise ValueError("In 'traditional' mode, 'model_path' must be provided.")
            
        ate_val, h_preds = _run_traditional_prediction(
            dataset=dataset,
            train_params=train_params, 
            model_path=model_dir,
            device=device
        )
        return {
            "POR": ate_val.item(),
            "h_preds": h_preds
        }
    
    train_params_copy = copy.deepcopy(train_params)
    base_params, search_space = _parse_optuna_space(train_params_copy)
    n_splits = int(base_params.get('n_splits', 5))
    
    if dataset is not None:
        ate_estimate, ate_values, h_preds = _run_cross_fitting(
            data_name=dataset_key,
            dataset=dataset,
            n_splits=n_splits,
            train_params=base_params,
            data_configs=data_configs,
            random_seed=random_seed,
            log_folder=log_folder,
            tag="external_cf",
        )
        return {
            "POR": ate_estimate.item(),
            "h_preds": h_preds
        }
    
    hpo_n_trials = int(base_params.pop('hpo_n_trials', 20))
    final_train_params: Dict[str, Any] = {}

    
    if dataset_key == 'sgd':
        data_base_path = Path(data_configs['data_path'])
        n_trials = data_configs.get('n_trials',-1)
        n_samples = data_configs.get('n_samples',2000)
        scenario = data_configs.get('scenario', 1) 

        if n_trials > 0:
            POR = []
            use_u = base_params['use_u_statistic']
            loss_name = "u" if use_u == "true" else "v"
            for _ in range(1, n_trials + 1):
                df = generate_data(n_samples=n_samples, scenario= scenario)
                dataset_trial = SGDDataset(csv_path='', df=df, device=device)
                ate_estimate, ate_values, _= _run_cross_fitting(
                    data_name=dataset_key,
                    dataset=dataset_trial,
                    n_splits=n_splits,
                    train_params=base_params,
                    data_configs=data_configs,
                    random_seed=random_seed,
                    log_folder=log_folder,
                    tag="sgd_cf",
                )
                POR.append(ate_estimate.item())
            results = pd.DataFrame({
                "POR": POR
            })
            print(f"\n--- SGD ATE 估计结果 (平均 over {n_trials} 次试验):")
            print(f"POR = {results['POR'].mean():.4f}")
            print("----------------------------------")
            
            output_path = data_configs.get('output_path', 'predicts/sgd/nmmr_q')
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / f"por_{loss_name}_s{scenario}.csv"
            results.to_csv(out_file, index=False)
            
            
        else:
            train_df = generate_data(n_samples=n_samples, scenario=scenario)
            cf_df = generate_data(n_samples=n_samples, scenario=scenario)

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
            if search_space:
                print("\n===> 开始 SGD 训练集 K-fold (Optuna 调参)")
                final_train_params = _run_optuna_tuning(
                    data_name = dataset_key,
                    dataset=base_dataset,
                    n_splits=n_splits,
                    n_trials=hpo_n_trials,
                    base_train_params=base_params,
                    search_space=search_space,
                    data_configs=data_configs,
                    random_seed=random_seed,
                )
            else:
                print("\n===> 使用固定默认参数")
                final_train_params = base_params
            
            final_train_params['n_epochs'] += 200

            print("\n===> 在 SGD CF 数据集上执行 cross fitting (使用最优/固定参数)")
            ate_estimate, ate_values, _ = _run_cross_fitting(
                data_name=dataset_key,
                dataset=cf_dataset,
                n_splits=n_splits,
                train_params=final_train_params,
                data_configs=data_configs,
                random_seed=random_seed,
                log_folder=log_folder,
                tag="sgd_test",
            )

    elif dataset_key == 'rhc':
        rhc_data_dir = data_configs.get('data_path')
        use_all_x = data_configs.get('use_all_X', False)
        trainer_data_configs = {'rhc': data_configs}

        print(f"加载 RHC 数据集 (use_all_X={use_all_x})...")
        

        train_dataset = RHCDataset(split='train', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)

        if search_space:
            print("\n===> [RHC] 开始训练集 K-fold HPO")
            final_train_params = _run_optuna_tuning(
                dataset=train_dataset,
                n_splits=n_splits,
                n_trials=hpo_n_trials,
                base_train_params=base_params,
                search_space=search_space,
                data_configs=trainer_data_configs,
                random_seed=random_seed,
            )
        else:
            print("\n===> [RHC] 跳过 HPO，使用默认参数")
            final_train_params = base_params

        print(f"\n===> [RHC] 加载 Val/Test 集并合并，准备执行 Cross Fitting")
        val_dataset = RHCDataset(split='val', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
        test_dataset = RHCDataset(split='test', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)

        full_dataset = MergedDataset([train_dataset, val_dataset, test_dataset])
        print(f"     全量数据样本数: {len(full_dataset)}")

        ate_estimate, ate_values = _run_cross_fitting(
            data_name=dataset_key,
            dataset=full_dataset,
            n_splits=n_splits,
            train_params=final_train_params,
            data_configs=trainer_data_configs,
            random_seed=random_seed,
            log_folder=log_folder,
            tag="rhc_full",
        )

    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    print(f"\n--- 最终 ATE(POR) (Seed {random_seed}) ---")
    print(f"ATE(POR) = {ate_estimate.item():.4f}")
    print("----------------------------------")

def train_h_standalone(data_configs, train_params, dataset_name, scenario, save_dir, log_folder, random_seed=42):
    
    """
    独立训练模式：
    1. 加载/生成数据
    2. (可选) 使用 Optuna 进行 K-Fold 调参
    3. 使用最佳参数在全量数据上训练
    4. 保存模型权重和最佳参数配置
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Standalone_h] 启动独立训练流程 | 数据集: {dataset_name} | Scenario: {scenario}")
    
    dataset_key = dataset_name.lower()
    if dataset_key == 'sgd':
        n_samples = data_configs.get('n_samples',2000)
        df = generate_data(n_samples=n_samples, scenario=scenario)
        train_dataset = SGDDataset(csv_path='', df=df, device=device)
        
    elif dataset_key == 'rhc':
        print(f"[Standalone] 加载 RHC 训练数据...")
        data_dir = data_configs.get('data_path')
        use_all_x = data_configs.get('use_all_X', False)
        train_dataset = RHCDataset(split='train', use_all_x=use_all_x, data_dir=data_dir, device=device)
    else:
        raise ValueError(f"未知的 dataset_name: {dataset_name}")
    
    
    train_params_copy = copy.deepcopy(train_params)
    base_params, search_space = _parse_optuna_space(train_params_copy)
    n_splits = int(base_params.get('n_splits', 5))

    hpo_n_trials = int(base_params.pop('hpo_n_trials', 20))  

    _run_optuna_tuning(
        data_name=dataset_key,
        dataset=train_dataset,
        n_splits=n_splits,
        n_trials=hpo_n_trials,
        base_train_params=base_params,
        search_space=search_space,
        data_configs=data_configs,
        random_seed=random_seed,
        log_folder=log_folder,
        save_dir=save_dir,      
        scenario=scenario       
    )