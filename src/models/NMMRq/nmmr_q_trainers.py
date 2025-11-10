import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from src.data.ate.data_class import SGDDataset, RHCDataset
# from torch.utils.data import DataLoader
from src.models.NMMRq.nmmr_q_loss import NMMR_Q_Loss
from src.models.NMMRq.nmmr_q_model import NMMR_Q_common
# from src.models.NMMRq.kernel_utils import calculate_kernel_matrix_batched


def _resolve_dataset(dataset):
    """
    DataLoader 在 K-Fold 下会传入 Subset，这里统一解包。
    返回底层数据集以及可选的索引列表。
    """
    if isinstance(dataset, Subset):
        return dataset.dataset, dataset.indices
    return dataset, None


def _select_rows(tensor, indices):
    """
    根据给定索引切片张量；如果 indices 为空则直接返回原张量。
    """
    if indices is None:
        return tensor
    return tensor[indices]


class NMMR_Q_Trainer_SGD:
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], random_seed: int,
                 dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.train_params = train_params
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = torch.cuda.is_available() 
        self.log_metrics = train_params['log_metrics'] == "True"
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']
        self.mse_loss = nn.MSELoss()
        '''
        记录设备信息并实例化 q-损失，确保与 kernel/loss 配置一致
        '''
        self.device = torch.device('cuda' if self.gpu_flg else 'cpu')
        self.q_loss = NMMR_Q_Loss(
            kernel_gamma=self.train_params.get('kernel_gamma', 1.0),
            use_u_statistic=self.train_params.get('use_u_statistic', False),
            device=self.device.type,
        )
        
        self.writer = None
        if self.log_metrics and dump_folder is not None:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []
        else:
            self.log_metrics = False
    # def compute_kernel(self, kernel_inputs):
    #     return calculate_kernel_matrix_batched(kernel_inputs)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, verbose: int = 0) -> NMMR_Q_common:
        """
        训练NMMR模型。
        
        参数:
        train_loader (DataLoader): 包含 SGDDataset 的训练数据加载器。
        val_loader (DataLoader): 包含 SGDDataset 的验证数据加载器。
        verbose (int): 日志详细程度 (0 = 安静)。
        
        返回:
        NMMR_Q_common: 训练好的模型。
        """
        
        # 从DataLoader访问底层的Dataset
        train_dataset, train_indices = _resolve_dataset(train_loader.dataset)
        val_dataset, val_indices = _resolve_dataset(val_loader.dataset)

        # --- 自动计算输入维度 ---
        # 模型的输入是 A, Z, X
        A_dim = train_dataset.A.shape[1]
        Z_dim = train_dataset.Z.shape[1]
        X_dim = train_dataset.X.shape[1]
        input_size = A_dim + Z_dim + X_dim
        # -----------------------------

        model = NMMR_Q_common(input_dim=input_size, train_params=self.train_params)

        # 假设: SGDDataset 已经将数据放到了正确的设备上 (cpu or cuda)
        # 我们只需要将模型移动到该设备
        if self.gpu_flg:
            model.cuda()

        # weight_decay 实现 L2 惩罚
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        print(f"开始训练 NMMR_Q模型, 输入维度: {input_size}")
        
        # 训练模型
        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            model.train() # 设置模型为训练模式
            
            # DataLoader 循环
            for batch_data in train_loader:
                
                # --- 从字典中解包数据 ---
                # 假设数据已在 SGDDataset 的 __init__ 中被移动到正确设备
                '''
                批次迁移到目标设备，确保与模型一致
                '''
                batch_A = batch_data['A'].to(self.device)
                batch_W = batch_data['W'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_X = batch_data['X'].to(self.device)
                # batch_y = batch_data['Y'].to(self.device)
                # -----------------------------

                optimizer.zero_grad()
                
                # 模型输入: (A, Z, X)
                batch_inputs = torch.cat((batch_A, batch_Z, batch_X), dim=1)
                pred_y = model(batch_inputs)
                '''
                使用自定义 q-loss 计算梯度
                '''
                causal_loss_train, _ = self.q_loss(
                    q_a_hat=pred_y,
                    a=batch_A,
                    w=batch_W,
                    x=batch_X,
                )
                causal_loss_train.backward()
                optimizer.step()

            # 在每个 epoch 结束时，记录指标
            if self.log_metrics:
                model.eval() # 设置模型为评估模式
                with torch.no_grad():
                    # --- 在整个训练集上评估 ---
                    '''
                    全量训练集迁移到设备做度量
                    '''
                    train_A = _select_rows(train_dataset.A, train_indices).to(self.device)
                    train_W = _select_rows(train_dataset.W, train_indices).to(self.device)
                    train_Z = _select_rows(train_dataset.Z, train_indices).to(self.device)
                    train_X = _select_rows(train_dataset.X, train_indices).to(self.device)
                    train_Y = _select_rows(train_dataset.Y, train_indices).to(self.device)
                    preds_train = model(torch.cat((train_A, train_Z, train_X), dim=1))
                    
                    # "观测" MSE (非因果)
                    mse_train = self.mse_loss(preds_train, train_Y)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    
                    # 因果损失
                    causal_loss_train_full, _ = self.q_loss(
                        q_a_hat=preds_train,
                        a=train_A,
                        w=train_W,
                        x=train_X,
                    )
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train_full, epoch)
                    self.causal_train_losses.append(causal_loss_train_full.item()) # .item()
                    
                    # --- 在整个验证集上评估 ---
                    '''
                    全量验证集迁移到设备做度量
                    '''
                    val_A = _select_rows(val_dataset.A, val_indices).to(self.device)
                    val_W = _select_rows(val_dataset.W, val_indices).to(self.device)
                    val_Z = _select_rows(val_dataset.Z, val_indices).to(self.device)
                    val_X = _select_rows(val_dataset.X, val_indices).to(self.device)
                    val_Y = _select_rows(val_dataset.Y, val_indices).to(self.device)
                    preds_val = model(torch.cat((val_A, val_Z, val_X), dim=1))
                    
                    # "观测" MSE (非因果)
                    mse_val = self.mse_loss(preds_val, val_Y)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # 因果损失
                    causal_loss_val_full, _ = self.q_loss(
                        q_a_hat=preds_val,
                        a=val_A,
                        w=val_W,
                        x=val_X,
                    )
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val_full, epoch)
                    self.causal_val_losses.append(causal_loss_val_full.item()) # .item()

        print("训练完成。")
        return model

    @staticmethod
    def predict(model: NMMR_Q_common, test_dataset: 'SGDDataset'):
        """
        按照 φ̂_U(V) = (1/n) ∑ (-1)^{1-a_i} q̂_U(V)(a_i, x_i, z_i) y_i 的形式
        对给定数据集计算估计值。
        """
        model.eval()

        model_device = next(model.parameters()).device
        A_samples = test_dataset.A.to(model_device)
        Z_samples = test_dataset.Z.to(model_device)
        X_samples = test_dataset.X.to(model_device)
        Y_samples = test_dataset.Y.to(model_device)

        inputs = torch.cat((A_samples, Z_samples, X_samples), dim=1)

        with torch.no_grad():
            q_hat = model(inputs)

        # (-1)^{1-a_i}: a=1 -> 1, a=0 -> -1
        signs = torch.where(A_samples > 0.5, torch.ones_like(A_samples), -torch.ones_like(A_samples))
        phi_hat = (signs * q_hat * Y_samples).mean()

        return phi_hat.cpu()
        
