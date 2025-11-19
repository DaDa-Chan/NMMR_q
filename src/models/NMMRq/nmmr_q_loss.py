import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 自定义损失函数类 NMMR_Q_Loss
# ---------------------------------------------------------------------------
class NMMR_Q_Loss(nn.Module):
    
    
    def __init__(self, 
                 use_u_statistic=False):
        super(NMMR_Q_Loss, self).__init__()
        self.use_u_statistic = use_u_statistic

    def forward(self, q_pred_sub, mask_sub, kernel_matrix):

        
        N = kernel_matrix.shape[0]
        
        residuals = torch.full((N, 1), -1.0, device=kernel_matrix.device, dtype=kernel_matrix.dtype)
        
        # 更新目标组 (A = a) 的残差为 q - 1.0
        # 注意: q_pred_sub 形状需为 [N_sub, 1]
        if mask_sub.sum() > 0:
            residuals[mask_sub] = q_pred_sub - 1.0

        
        if self.use_u_statistic:

            kernel_matrix_na = kernel_matrix.fill_diagonal_(0)
            loss_sum = residuals.T @ kernel_matrix_na @ residuals

            if N > 1:
                loss_val = loss_sum / (N * (N - 1))
            else:
                loss_val = torch.tensor(0.0, device=kernel_matrix.device)
                
        else:

            loss_sum = residuals.T @ kernel_matrix @ residuals
            loss_val = loss_sum / (N * N)
            
        return loss_val