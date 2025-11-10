import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.models.NMMRq.nmmr_q_trainers import NMMR_Q_Trainer_SGD
from src.data.ate.data_class import SGDDataset, RHCDataset


def _locate_split_file(
    base_path: Path,
    explicit_name: Optional[str],
    fallback_names: List[str],
) -> Optional[Path]:
    """
    Helper to locate a split file. Tries explicit name first,
    then a list of fallback names. Returns None if nothing exists.
    """
    candidates: List[str] = []
    if explicit_name:
        candidates.append(explicit_name)
    candidates.extend(fallback_names)

    for name in candidates:
        candidate_path = base_path / name
        if candidate_path.exists():
            return candidate_path
    return None


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
    n_splits = int(train_params.get('n_splits', 5))

    if dataset_key == 'sgd':
        data_base_path = Path(data_configs['data_path'])
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

        print("\n===> 开始 SGD 训练集 K-fold 交叉验证")
        _run_cross_validation(
            dataset=base_dataset,
            n_splits=n_splits,
            train_params=train_params,
            data_configs=data_configs,
            random_seed=random_seed,
            dump_folder=None,
        )

        print("\n===> 在 SGD CF 数据集上执行 cross fitting")
        ate_estimate, ate_values = _run_cross_fitting(
            dataset=cf_dataset,
            n_splits=n_splits,
            train_params=train_params,
            data_configs=data_configs,
            random_seed=random_seed,
            dump_folder=None,
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


def _run_cross_validation(dataset,
                          n_splits: int,
                          train_params: Dict[str, Any],
                          data_configs: Dict[str, Any],
                          random_seed: int,
                          dump_folder: Optional[Path]):
    """
    在给定数据集上执行 K-fold 交叉验证，仅用于 SGD 训练集。
    """
    batch_size = train_params['batch_size']
    cv_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    val_losses: List[float] = []
    best_val = float('inf')

    for fold_idx, (train_idx, val_idx) in enumerate(cv_kfold.split(np.arange(len(dataset))), start=1):
        print(f"\n[CV] Fold {fold_idx}/{n_splits}")
        fold_folder = None
        if dump_folder is not None:
            fold_folder = dump_folder / f"fold_{fold_idx}"
            fold_folder.mkdir(parents=True, exist_ok=True)

        trainer = _create_trainer(data_configs, train_params, random_seed, fold_folder)

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

        model = trainer.train(train_loader, val_loader)
        val_loss = _evaluate_q_loss(trainer, model, dataset, val_idx.tolist())
        val_losses.append(val_loss)
        best_val = min(best_val, val_loss)
        print(f"[CV] Fold {fold_idx} q-loss: {val_loss:.6f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if val_losses:
        print(f"\n[CV] 平均验证损失: {np.mean(val_losses):.6f} | 最优: {best_val:.6f}")
    else:
        print("[CV] 未能计算验证损失，请检查配置。")


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

        # model_save_path = fold_folder / f"trained_model_seed_{random_seed}_fold_{fold_idx}.pth"
        # try:
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"[CF-{tag}] 模型保存于: {model_save_path}")
        # except Exception as e:
        #     print(f"[CF-{tag}] 警告: 无法保存模型。错误: {e}")

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
        inputs = torch.cat((A, W, X), dim=1)
        preds = model(inputs)
        loss, _ = trainer.q_loss(q_a_hat=preds, a=A, w=W, x=X)
    return float(loss.item())
