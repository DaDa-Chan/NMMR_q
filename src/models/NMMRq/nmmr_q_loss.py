import torch
import torch.nn as nn
from src.models.NMMRq.kernel_utils import (
    G_kernel,
    calculate_kernel_matrix_batched,
)

# ---------------------------------------------------------------------------
# 2. 修改后的损失函数类 NMMR_Q_Loss
# ---------------------------------------------------------------------------
class NMMR_Q_Loss(nn.Module):
    
    
    def __init__(self, 
                 kernel_gamma=1.0, 
                 use_u_statistic=False,
                 device='cpu'):
        """
        初始化损失函数
        
        参数:
        kernel_gamma (float): RBF 核 k_w 的 gamma 参数。
        use_u_statistic (bool): 
            - False (默认): 使用 V-statistic，包含对角线。
            - True: 使用 U-statistic，不包含对角线。
        device (str): 'cpu' 或 'cuda'
        """
        super(NMMR_Q_Loss, self).__init__()
        self.kernel_gamma = kernel_gamma
        self.use_u_statistic = use_u_statistic
        self.device = device

    def forward(self, q_a_hat, a, w, x):
        """
        计算损失函数的前向传播
        
        参数:
        q_a_hat (torch.Tensor): q 网络的输出 q(Z, A, X)，形状 [batch_size, 1]
        a (torch.Tensor): 处理变量，形状 [batch_size, 1]
        z (torch.Tensor): 结局代理变量，形状 [batch_size, z_dim]
        w (torch.Tensor): 处理代理变量，形状 [batch_size, w_dim]
        x (torch.Tensor): 协变量，形状 [batch_size, x_dim]
        
        返回:
        torch.Tensor: 计算得到的标量损失值
        dict: 包含损失详情的字典
        """
        
        total_loss = 0.0
        
        # 我们需要分别计算 A=0 和 A=1 时的损失，然后相加
        for a_val in [0.0, 1.0]:
            
            # 1. 筛选出当前处理组 (A=a) 的数据
            indices = (a == a_val).squeeze()
            
            # 如果当前批次中没有这个组的数据，跳过
            if indices.sum() < 2:  # U-statistic 至少需要2个样本
                continue
                
            # 提取对应组的数据
            q_group = q_a_hat[indices]  # 形状 [Na, 1]
            w_group = w[indices]        # 形状 [Na, z_dim]
            x_group = x[indices]        # 形状 [Na, x_dim]
            
            Na = q_group.shape[0]

            # 2. 准备公式 (13) 中的各项
            
            # 2.a. 计算 (q_a(Z_i, X_i) - 1)
            q_minus_1 = q_group - 1.0  # 形状 [Na, 1]
                        
            # 2.b. 计算核矩阵 k_w,ij
            wx_group = torch.cat([w_group, x_group], dim=1)  # 形状 [Na, D]
                       
            k_w_matrix = calculate_kernel_matrix_batched(
                dataset=wx_group,
                batch_indices=(0, Na),
                kernel=G_kernel,
                gamma=self.kernel_gamma,
            )
            # 预期 k_w_matrix 的形状为 [Na, Na]
            # ---------------------------------------------------------------

            # 3. 计算最终的损失值 (V-statistic 或 U-statistic)
            
            # 矩阵中 (i, j) 元素 = (q_a(i)-1)(q_a(j)-1) * k_w(i,j)
            
            if self.use_u_statistic:
                # --- U-statistic (公式 15) ---
                # $\frac{1}{N_a(N_a - 1)} \sum_{i \neq j} ...$
                k_w_matrix = k_w_matrix - k_w_matrix.diag()
                loss_sum = q_minus_1.T @ k_w_matrix @ q_minus_1
                loss_val = loss_sum / (Na * (Na - 1))
            else:
                # --- V-statistic (公式 14) ---
                # $\frac{1}{N_a^2} \sum_{i, j} ...$
                loss_sum = q_minus_1.T @ k_w_matrix @ q_minus_1
                loss_val = loss_sum / (Na * Na)
            
            # 累加 A=0 和 A=1 的损失
            total_loss += loss_val
        
        loss_dict = {
            'q_moment_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }
        
        return total_loss, loss_dict
