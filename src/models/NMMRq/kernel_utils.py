import numpy as np
import torch

def fit_sigma(data: torch.Tensor, scale: float = 1.0) -> float:
    """
    Args:
        data: 用于拟合的数据集 (e.g., 训练数据)
              (形状: [n_samples, n_features])
        scale: 缩放因子 

    Returns:
        计算得到的 sigma。
    """
    # 1. 计算所有样本间的两两欧几里得距离
    #    使用 torch.pdist 获取一个包含 N*(N-1)/2 个距离的向量，
    dists = torch.pdist(data, p=2.0)
    
    # 2. 找到距离的中位数
    #    + 1e-5 是为了防止中位数为0导致除零错误
    median_dist = torch.median(dists) + 1e-5

    # 3. 计算 sigma
    sigma = 2.0 * (median_dist ** 2) * scale
    
    return sigma.item()

def G_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    gamma: float | None = None,
):
    """
    Gaussian(RBF) kernel. 提供 length_scale 或 gamma(=1/(2l^2)) 以复用同一实现。
    """
    squared_distance = torch.sum((x - y) ** 2, dim=0)
    if gamma is not None:
        return torch.exp(-gamma * squared_distance)
    denom = sigma
    return torch.exp(-squared_distance / denom)

def calculate_kernel_matrix(dataset, kernel=G_kernel, **kargs):
    tensor = dataset.permute(1, 0)
    tensor1 = tensor.unsqueeze(dim=2)
    tensor2 = tensor.unsqueeze(dim=1)
    
    return kernel(tensor1, tensor2, **kargs)

def calculate_kernel_matrix_batched(dataset, batch_indices:tuple, kernel=G_kernel, **kwargs):
    tensor = dataset.permute(1,0)
    tensor1 = tensor.unsqueeze(dim=2)
    tensor1 = tensor1[:, batch_indices[0]:batch_indices[1], :]
    tensor2 = tensor.unsqueeze(dim=1)

    return kernel(tensor1, tensor2, **kwargs)
