import numpy as np
import torch

def G_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    length_scale: float = 1.0,
    gamma: float | None = None,
):
    """
    Gaussian(RBF) kernel. 提供 length_scale 或 gamma(=1/(2l^2)) 以复用同一实现。
    """
    squared_distance = torch.sum((x - y) ** 2, dim=0)
    if gamma is not None:
        return torch.exp(-gamma * squared_distance)
    denom = 2 * (length_scale ** 2)
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
