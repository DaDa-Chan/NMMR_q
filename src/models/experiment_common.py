from dataclasses import dataclass, field

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def to_bool(value: Any, default: bool = False) -> bool:
    """
    Robustly parse a flag that may come from a JSON config as a real
    bool, or (for configs not yet migrated) as a "True"/"False" string.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def check_loss_name_matches_u_statistic(train_params: Dict[str, Any]) -> None:
    """
    Guard against a config where `loss_name` and `use_u_statistic` disagree
    about whether this run is U-statistic or V-statistic (e.g. loss_name=
    "U_statistic" but use_u_statistic=false). Raises on mismatch instead of
    silently training with one and labeling/logging with the other.
    """
    loss_name = train_params.get('loss_name')
    use_u_statistic = to_bool(train_params.get('use_u_statistic', False))
    if loss_name is None:
        return
    loss_name_lower = str(loss_name).strip().lower()
    if loss_name_lower not in ("u_statistic", "v_statistic"):
        return
    expects_u = loss_name_lower == "u_statistic"
    if expects_u != use_u_statistic:
        raise ValueError(
            f"Config inconsistency: loss_name={loss_name!r} implies "
            f"use_u_statistic={expects_u}, but use_u_statistic={use_u_statistic} was given."
        )


@dataclass
class TrainResult:
    """
    Structured outcome of one Trainer.train() call, returned alongside the
    trained model so callers (cross-fitting, HPO, standalone training) can
    inspect how training actually went instead of parsing console output.
    """
    best_epoch: int = 0
    best_val_score: Optional[float] = None
    last_epoch: int = 0
    stopped_early: bool = False
    stop_reason: str = "max_epochs"  # "max_epochs" | "early_stopping" | "no_valid_checkpoint"
    n_lr_reductions: int = 0
    final_learning_rate: Optional[float] = None
    train_seconds: float = 0.0
    best_state_restored: bool = False
    n_nonfinite_train: int = 0
    n_nonfinite_val: int = 0
    kernel_bandwidth: Optional[float] = None
    seed: Optional[int] = None


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
    trial, param_name: str, config: Dict[str, Any]
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


def _get_a_labels(dataset, indices) -> np.ndarray:
    """
    取出给定 indices 对应的处理变量 A,展平为一维 numpy 数组,供分层抽样使用。
    """
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return dataset.A[index_tensor].detach().cpu().numpy().reshape(-1)


def _stratified_kfold_split(dataset, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    按 dataset.A 分层的 K-fold 划分,用法与 KFold(...).split(...) 一致,
    返回 (train_idx, val_idx) 的列表。
    某个 A 取值样本数不足以分层时(每类样本数需 >= n_splits),回退到普通 KFold。
    """
    indices = np.arange(len(dataset))
    a_labels = _get_a_labels(dataset, indices)
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices, a_labels))
    except ValueError:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices))


def _build_loaders_for_fold(dataset,
                            indices: List[int],
                            batch_size: int,
                            seed: int):
    """
    将 fold 内部再划分为 train/val(按处理变量 A 分层抽样,保证 train/val 中
    A=0/A=1 的比例与原 fold 整体比例一致),以便 cross-fitting 阶段训练。
    某个 A 取值样本数过少无法分层时,回退到纯随机切分。
    """
    if len(indices) == 0:
        raise ValueError("fold 中没有样本。")

    indices_arr = np.array(indices)

    try:
        a_labels = _get_a_labels(dataset, indices_arr)
        train_idx, val_idx = train_test_split(
            indices_arr, test_size=0.2, random_state=seed, stratify=a_labels,
        )
    except ValueError:
        rng = np.random.RandomState(seed)
        shuffled = indices_arr.copy()
        rng.shuffle(shuffled)
        split = max(1, int(0.8 * len(shuffled)))
        train_idx = shuffled[:split]
        val_idx = shuffled[split:] if split < len(shuffled) else shuffled[:split]

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()

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
