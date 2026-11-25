import math

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
    # 计算所有样本间的两两欧几里得距离
    #    使用 torch.pdist 获取一个包含 N*(N-1)/2 个距离的向量，
    #    MPS 后端不支持 pdist(aten::_pdist_forward 未实现)，固定放 CPU 算：
    #    数据量只有 batch 大小，不是性能瓶颈，不影响主训练循环留在 MPS 上跑。
    dists = torch.pdist(data.detach().cpu(), p=2.0)

    # 找到距离的中位数
    #    + 1e-5 防止中位数为0导致除零错误
    median_dist = torch.median(dists) + 1e-5

    # 计算 sigma
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


def diag_weighted_loss(residuals, kernel_matrix, diag_tau):
    """
    Diagonal-weighted NMMR loss for a residual column vector r (shape [N, 1]):
        r^T K_off r / (N (N-1)) + tau * sum_i r_i^2 K_ii / N.
    tau = 0 reproduces the U-statistic. The input kernel matrix is not modified.
    Returns a [1, 1] tensor, matching the legacy U/V branches.
    """
    N = kernel_matrix.shape[0]
    k_diag = torch.diagonal(kernel_matrix)
    k_off = kernel_matrix.clone()
    k_off.fill_diagonal_(0)
    if N > 1:
        off_term = (residuals.T @ k_off @ residuals) / (N * (N - 1))
    else:
        off_term = torch.zeros((1, 1), device=kernel_matrix.device, dtype=kernel_matrix.dtype)
    diag_term = (residuals.squeeze(-1) ** 2 * k_diag).sum() / N
    return off_term + diag_tau * diag_term


# ---------------------------------------------------------------------------
# Kernel forms over blocks of kernel inputs (e.g. (A), (Z), (X) for h; (W), (X) for q)
# ---------------------------------------------------------------------------
#
#   "joint"        : one Gaussian kernel on the concatenated inputs with one
#                    median bandwidth. The trainers keep their original code
#                    path for this form, so existing results are unchanged.
#   "block_scaled" : each block is divided by sqrt(its dimension) before the
#                    concatenation, then one Gaussian kernel with one median
#                    bandwidth. The rescaling is fixed (it depends only on the
#                    block dimensions), so this is still a single-bandwidth
#                    Gaussian kernel on a fixed linear transform of the inputs.
#                    It keeps a high-dimensional block (many covariates) from
#                    swamping low-dimensional blocks (treatment, proxies).
#   "product"      : product over blocks: exact matching 1{a = a'} on blocks
#                    flagged exact (the binary treatment), and a Gaussian kernel
#                    with its own median bandwidth on every other block.

KERNEL_FORMS = ("joint", "block_scaled", "product")


def resolve_kernel_form(train_params) -> str:
    form = train_params.get("kernel_form", "joint")
    if form not in KERNEL_FORMS:
        raise ValueError(f"Unknown kernel_form {form!r}; expected one of {KERNEL_FORMS}.")
    return form


def _scaled_concat(blocks):
    return torch.cat([b / math.sqrt(b.shape[1]) for b in blocks], dim=1)


def fit_block_kernel(blocks, form: str, exact):
    """
    Bandwidths for a non-joint form, fitted once on training rows: one value
    for "block_scaled", one per non-exact block (None for exact blocks) for
    "product". Each is on the same scale as `fit_sigma` (the Gaussian
    denominator 2 * median^2).
    """
    if form == "block_scaled":
        return [fit_sigma(_scaled_concat(blocks))]
    if form == "product":
        return [None if ex else fit_sigma(b) for b, ex in zip(blocks, exact)]
    raise ValueError(f"fit_block_kernel does not handle kernel_form={form!r}.")


def block_kernel_matrix(blocks, form: str, sigmas, multiplier: float, exact):
    """Kernel matrix [N, N] of a non-joint form; `multiplier` scales every bandwidth."""
    n = blocks[0].shape[0]
    if form == "block_scaled":
        x = _scaled_concat(blocks)
        return calculate_kernel_matrix_batched(dataset=x, batch_indices=(0, n), kernel=G_kernel,
                                               sigma=sigmas[0] * multiplier, gamma=None)
    if form == "product":
        k = None
        for b, s, ex in zip(blocks, sigmas, exact):
            if ex:
                part = (b.reshape(-1, 1) == b.reshape(1, -1)).to(b.dtype)
            else:
                part = calculate_kernel_matrix_batched(dataset=b, batch_indices=(0, n), kernel=G_kernel,
                                                       sigma=s * multiplier, gamma=None)
            k = part if k is None else k * part
        return k
    raise ValueError(f"block_kernel_matrix does not handle kernel_form={form!r}.")
