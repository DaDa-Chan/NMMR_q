import torch
import torch.nn as nn

from src.models.kernel_utils import diag_weighted_loss

# ---------------------------------------------------------------------------
# 自定义损失函数类 NMMR_Q_Loss
# ---------------------------------------------------------------------------
class NMMR_Q_Loss(nn.Module):
    
    
    def __init__(self,
                 use_u_statistic=False,
                 diag_tau=None):
        """
        diag_tau: None keeps the legacy behaviour (use_u_statistic picks the
            U- or V-statistic). A float tau >= 0 switches to the
            diagonal-weighted loss
                r^T K_off r / (N (N-1)) + tau * sum_i r_i^2 K_ii / N,
            whose expectation over uniform mini-batches is R_U + tau * D
            (tau = 1/(n_train-1) gives n/(n-1) times the full-sample V risk).
            The trainer may set this attribute after construction, once the
            training-set size is known.
        """
        super(NMMR_Q_Loss, self).__init__()
        if diag_tau is not None and use_u_statistic:
            raise ValueError("diag_tau cannot be combined with use_u_statistic=True.")
        self.use_u_statistic = use_u_statistic
        self.diag_tau = diag_tau

    def forward(self, q_pred_sub, mask_sub, kernel_matrix):

        N = kernel_matrix.shape[0]

        residuals = torch.full((N, 1), -1, device=kernel_matrix.device, dtype=kernel_matrix.dtype)

        if mask_sub.sum() > 0:
            residuals[mask_sub] = q_pred_sub - 1.0

        if self.diag_tau is not None:
            return diag_weighted_loss(residuals, kernel_matrix, self.diag_tau)

        if self.use_u_statistic:

            kernel_matrix_na = kernel_matrix.fill_diagonal_(0)
            loss_sum = residuals.T @ kernel_matrix_na @ residuals
            
            if N > 1:
                loss_val = loss_sum / (N * (N - 1))
            else:
                loss_val = torch.tensor(0.0, device=kernel_matrix.device)
                
        else:
            # V-statistic: 1 / N^2 * sum_{i,j} rho_i * rho_j * K_ij
            loss_sum = residuals.T @ kernel_matrix @ residuals
            loss_val = loss_sum / (N * N)
            
        return loss_val