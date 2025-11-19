import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

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

class NMMR_Q_DualModel(nn.Module):
    
    def __init__(self, net0, net1):
        super().__init__()
        self.net0 = net0
        self.net1 = net1
    
    def forward(self, z, x, a=None):
        inputs = torch.cat((z, x), dim=1)
        pred0 = self.net0(inputs)
        pred1 = self.net1(inputs)
        
        if a is not None:
            return torch.where(a > 0.5, pred1, pred0)
        else:
            return pred0, pred1


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
        self.scaler = GradScaler() if self.gpu_flg else None
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
        训练NMMR_Q 模型。
        q0(Z, X) & q1(Z, X)
        
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
        # 模型的输入是 Z, X, A
        Z_dim = train_dataset.Z.shape[1]
        X_dim = train_dataset.X.shape[1]
        input_size = Z_dim + X_dim
        # -----------------------------

        model0 = NMMR_Q_common(input_dim=input_size, train_params=self.train_params)
        model1 = NMMR_Q_common(input_dim=input_size, train_params=self.train_params)
        # 假设: SGDDataset 已经将数据放到了正确的设备上 (cpu or cuda)
        # 我们只需要将模型移动到该设备
        if self.gpu_flg:
            model0.cuda()
            model1.cuda()

        # weight_decay 实现 L2 惩罚
        optimizer0 = optim.Adam(list(model0.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)
        optimizer1 = optim.Adam(list(model1.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        print(f"开始训练 NMMR_Q (q0, q1)模型, 输入维度: {input_size}")
        
        # 训练模型
        for epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            # 设置模型为训练模式
            model0.train() 
            model1.train()
            
            # DataLoader 循环
            for batch_data in train_loader:
                
                '''
                批次迁移到目标设备，确保与模型一致
                '''
                batch_A = batch_data['A'].to(self.device)
                batch_W = batch_data['W'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_X = batch_data['X'].to(self.device)

                # --- 分割数据分别训练 ---
                mask0 = (batch_A < 0.5).squeeze()
                if mask0.sum() > 1:
                    optimizer0.zero_grad()       
                    inputs0 = torch.cat((batch_Z[mask0], batch_X[mask0]), dim=1)
                    with autocast(enabled=self.gpu_flg): 
                        pred0 = model0(inputs0)
                        loss0, _ = self.q_loss(
                            q_a_hat=pred0,
                            a=batch_A[mask0],
                            w=batch_W[mask0],
                            x=batch_X[mask0]
                        )
                    if self.scaler:
                        self.scaler.scale(loss0).backward()
                        self.scaler.step(optimizer0)
                        self.scaler.update()
                    else:                        
                        loss0.backward()
                        optimizer0.step()

                mask1 = (batch_A > 0.5).squeeze()
                if mask1.sum() > 1:
                    optimizer1.zero_grad()
                    inputs1 = torch.cat((batch_Z[mask1], batch_X[mask1]), dim=1)
                    with autocast(enabled=self.gpu_flg): 
                        pred1 = model1(inputs1)
                        loss1, _ = self.q_loss(
                            q_a_hat=pred1,
                            a=batch_A[mask1],
                            w=batch_W[mask1],
                            x=batch_X[mask1]
                        )
                    if self.scaler:
                        self.scaler.scale(loss1).backward()
                        self.scaler.step(optimizer1)
                        self.scaler.update()
                    else:                        
                        loss1.backward()
                        optimizer1.step()
                

            # 在每个 epoch 结束时，记录指标
            if self.log_metrics:
                model0.eval() 
                model1.eval()
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
                    
                    inputs_train = torch.cat((train_Z, train_X), dim=1)
                    out0_train = model0(inputs_train)
                    out1_train = model1(inputs_train)
                    preds_train = torch.where(train_A > 0.5, out1_train, out0_train)
                    
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
                    
                    inputs_val = torch.cat((val_Z, val_X), dim=1)
                    out0_val = model0(inputs_val)
                    out1_val = model1(inputs_val)
                    preds_val = torch.where(val_A > 0.5, out1_val, out0_val)
                    

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
        return NMMR_Q_DualModel(model0, model1)

    @staticmethod
    def predict(model: NMMR_Q_DualModel, dataset_view):
        """
        按照 φ̂_U(V) = (1/n) ∑ (-1)^{1-a_i} q̂_U(V)(a_i, x_i, z_i) y_i 的形式
        对给定数据集计算估计值。
        """
        model.eval()

        model_device = next(model.parameters()).device
        A_samples = dataset_view.A.to(model_device)
        Z_samples = dataset_view.Z.to(model_device)
        X_samples = dataset_view.X.to(model_device)
        Y_samples = dataset_view.Y.to(model_device)

        #inputs_A0 = torch.cat((Z_samples, X_samples, torch.zeros_like(A_samples)), dim=1)
        #inputs_A1 = torch.cat((Z_samples, X_samples, torch.ones_like(A_samples)), dim=1)
        
        with torch.no_grad():
            #q_hat_0 = model(inputs_A0)
            #q_hat_1 = model(inputs_A1)
            q_hat = model(Z_samples, X_samples, A_samples)

        # (-1)^{1-a_i}: a=1 -> 1, a=0 -> -1
        signs = torch.where(A_samples > 0.5, torch.ones_like(A_samples), -torch.ones_like(A_samples))
        #phi_hat_counter = (q_hat_1 * Y_samples - q_hat_0 * Y_samples).mean()
        phi_hat = (signs * q_hat * Y_samples).mean()

        return phi_hat.cpu()
        
