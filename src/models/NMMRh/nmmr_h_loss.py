import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 自定义损失函数类 NMMR_H_Loss
# ---------------------------------------------------------------------------
class NMMR_H_Loss(nn.Module):
    
    def __init__(self, 
                 use_u_statistic=False):
        """
        初始化 NMMR H Loss (求解 h 函数)
        
        参数:
        use_u_statistic (bool): 
            - False (默认): 使用 V-statistic (1/N^2 * sum)
            - True: 使用 U-statistic (1/(N*(N-1)) * sum, i!=j)
        """
        super(NMMR_H_Loss, self).__init__()
        self.use_u_statistic = use_u_statistic

    def forward(self, model_output, target, kernel_matrix):
        """
        前向传播计算 Loss
        
        参数:
        model_output: 模型预测值 h(A, W, X), shape [N, 1]
        target: 真实值 Y, shape [N, 1]
        kernel_matrix: 核矩阵 K, shape [N, N]
        """
        
        # 1. 计算残差 (保持原公式逻辑: target - output)
        residual = target - model_output
        n = residual.shape[0]
        
        # 2. 根据统计量类型计算 Loss
        if self.use_u_statistic:
            # U-statistic: 1 / (n * (n-1)) * r^T * K_no_diag * r
            # 为了不影响反向传播和原矩阵，建议先 clone 再修改对角线
            K = kernel_matrix.clone()
            K.fill_diagonal_(0)
            
            loss_sum = residual.T @ K @ residual
            
            if n > 1:
                loss_val = loss_sum / (n * (n - 1))
            else:
                loss_val = torch.tensor(0.0, device=kernel_matrix.device)
                
        else:
            # V-statistic: 1 / n^2 * r^T * K * r
            loss_sum = residual.T @ kernel_matrix @ residual
            loss_val = loss_sum / (n ** 2)
            
        # 返回标量 (去除多余维度 [1, 1] -> scalar)
        return loss_val.squeeze()