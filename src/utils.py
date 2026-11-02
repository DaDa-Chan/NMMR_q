import random
from dataclasses import dataclass

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """
    显式设置 torch/numpy/random 的全局种子。
    用作某个独立种子角色(如 init_seed)在各随机数生成库上的落地点，
    不在本函数之外直接调用 torch.manual_seed/np.random.seed/random.seed。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@dataclass
class SeedBundle:
    """一次重复(repetition)内五个独立随机性角色的种子。"""
    data_seed: int
    split_seed: int
    init_seed: int
    tuning_seed: int
    bootstrap_seed: int


def make_seed_bundle(*entropy: int) -> "SeedBundle":
    """
    从任意数量的整数熵值(如 run_seed、scenario、trial_index)派生出
    data_seed/split_seed/init_seed/tuning_seed/bootstrap_seed 五个独立种子。

    使用 numpy SeedSequence.spawn 保证各角色统计独立，且派生规则固定、
    只依赖传入的 entropy，不依赖调用顺序或任何全局可变状态，因此可复现。
    """
    seq = np.random.SeedSequence(list(entropy))
    data_seq, split_seq, init_seq, tuning_seq, bootstrap_seq = seq.spawn(5)
    to_int = lambda s: int(s.generate_state(1)[0])
    return SeedBundle(
        data_seed=to_int(data_seq),
        split_seed=to_int(split_seq),
        init_seed=to_int(init_seq),
        tuning_seed=to_int(tuning_seq),
        bootstrap_seed=to_int(bootstrap_seq),
    )


def get_device() -> torch.device:
    """
    Select the computing device according to priority: CUDA > Apple Silicon MPS > CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def empty_device_cache(device: torch.device) -> None:
    """Release the cached memory on the corresponding device."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
