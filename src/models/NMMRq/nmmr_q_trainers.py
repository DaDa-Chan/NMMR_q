import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path
import copy
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


from src.models.NMMRq.nmmr_q_loss import NMMR_Q_Loss
from src.models.NMMRq.nmmr_q_model import NMMR_Q_common, NMMR_Q_DualModel
from src.models.NMMRq.kernel_q_utils import fit_sigma, G_kernel, calculate_kernel_matrix_batched


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

class EarlyStopping:
    """
    当验证集 loss 在 patience 个 epoch 内没有提升时停止训练
    """
    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience (int): 上次 loss 改善后等待的 epoch 数。默认 10。
            min_delta (float): 被视为提升的最小变化量。小于此值的变化被视为无提升。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model0_state = None
        self.best_model1_state = None

    def __call__(self, val_loss, model0, model1):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model0, model1)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss 下降了
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model0, model1)
            self.counter = 0

    def save_checkpoint(self, val_loss, model0, model1):
        '''保存当前最佳模型参数'''
        self.best_model0_state = copy.deepcopy(model0.state_dict())
        self.best_model1_state = copy.deepcopy(model1.state_dict())

class NMMR_Q_Trainer:
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
        self.scaler = torch.amp.GradScaler('cuda') if self.gpu_flg else None
        self.kernel_gamma = train_params['kernel_gamma']
        self.device = torch.device('cuda' if self.gpu_flg else 'cpu')
        self.q_loss = NMMR_Q_Loss(
            use_u_statistic=self.train_params.get('use_u_statistic', False),
        )
        
        self.writer = None
        if self.log_metrics and dump_folder is not None:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []
        else:
            self.log_metrics = False

    def _compute_kernel_matrix(self, w, x):
        """
        辅助函数：计算全 Batch 的核矩阵
        输入: w [N, w_dim], x [N, x_dim]
        输出: K [N, N]
        """
        N = w.shape[0]
        wx_group = torch.cat([w, x], dim=1)  # [N, D]
        sigma_data = fit_sigma(wx_group)
        
        # 使用 kernel_utils 中的函数计算全矩阵
        k_matrix = calculate_kernel_matrix_batched(
            dataset=wx_group,
            batch_indices=(0, N),
            kernel=G_kernel,
            sigma=sigma_data,
            gamma=self.kernel_gamma,
        )
        return k_matrix
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, verbose: int = 0) -> NMMR_Q_common:
        """
        训练NMMR_Q 模型。
        q0(Z, X) & q1(Z, X)
        """

        train_dataset, train_indices = _resolve_dataset(train_loader.dataset)
        val_dataset, val_indices = _resolve_dataset(val_loader.dataset)

        Z_dim = train_dataset.Z.shape[1]
        X_dim = train_dataset.X.shape[1]
        input_size = Z_dim + X_dim
        # -----------------------------

        model0 = NMMR_Q_common(input_dim=input_size, train_params=self.train_params)
        model1 = NMMR_Q_common(input_dim=input_size, train_params=self.train_params)

        if self.gpu_flg:
            model0.cuda()
            model1.cuda()

        optimizer0 = optim.Adam(list(model0.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)
        optimizer1 = optim.Adam(list(model1.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)
        
        sched_patience = self.train_params.get('scheduler_patience', 5)
        sched_factor = self.train_params.get('scheduler_factor', 0.5)
        
        scheduler0 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer0, mode='min', factor=sched_factor, patience=sched_patience, min_lr=1e-6
        )
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, mode='min', factor=sched_factor, patience=sched_patience, min_lr=1e-6
        )
 
        patience = self.train_params.get('patience', 20) 
        early_stopping = EarlyStopping(patience=patience, min_delta=5e-5)

        for epoch in tqdm(range(self.n_epochs), desc="Epochs", disable=not verbose):

            model0.train() 
            model1.train()
            
            
            for batch_data in train_loader:
                
                '''
                批次迁移到目标设备，确保与模型一致
                '''
                batch_A = batch_data['A'].to(self.device)
                batch_W = batch_data['W'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_X = batch_data['X'].to(self.device)

                with torch.no_grad():
                    kernel_matrix = self._compute_kernel_matrix(batch_W, batch_X)
                

                mask0 = (batch_A < 0.5).squeeze()
                if mask0.sum() > 1:
                    optimizer0.zero_grad()       
                    inputs0 = torch.cat((batch_Z[mask0], batch_X[mask0]), dim=1)
                    with torch.amp.autocast(device_type='cuda', enabled=self.gpu_flg):
                        pred0 = model0(inputs0)
                        loss0 = self.q_loss(
                            q_pred_sub=pred0,
                            mask_sub=mask0,
                            kernel_matrix=kernel_matrix
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
                    with torch.amp.autocast(device_type='cuda', enabled=self.gpu_flg):
                        pred1 = model1(inputs1)
                        loss1 = self.q_loss(
                            q_pred_sub=pred1,
                            mask_sub=mask1,
                            kernel_matrix=kernel_matrix
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
                    
                    k_train = self._compute_kernel_matrix(train_W, train_X)
            
                    mask0_train = (train_A < 0.5).squeeze()
                    if mask0_train.sum() > 0:
                        input0 = torch.cat((train_Z[mask0_train], train_X[mask0_train]), dim=1)
                        p0 = model0(input0)
                        loss0_train = self.q_loss(p0, mask0_train, k_train)
                    else:
                        loss0_train = 0.0

                    mask1_train = (train_A > 0.5).squeeze()
                    if mask1_train.sum() > 0:
                        input1 = torch.cat((train_Z[mask1_train], train_X[mask1_train]), dim=1)
                        p1 = model1(input1)
                        loss1_train = self.q_loss(p1, mask1_train, k_train)
                    else:
                        loss1_train = 0.0
                    
                    # 因果损失
                    total_train_loss = loss0_train + loss1_train
                    
                    self.writer.add_scalar(f'{self.loss_name}/train', total_train_loss, epoch)
                    self.causal_train_losses.append(total_train_loss.item()) # .item()
                    
                    # --- 在整个验证集上评估 ---
                    '''
                    全量验证集迁移到设备做度量
                    '''
                    val_A = _select_rows(val_dataset.A, val_indices).to(self.device)
                    val_W = _select_rows(val_dataset.W, val_indices).to(self.device)
                    val_Z = _select_rows(val_dataset.Z, val_indices).to(self.device)
                    val_X = _select_rows(val_dataset.X, val_indices).to(self.device)
                    
                    k_val = self._compute_kernel_matrix(val_W, val_X)
                    
                    mask0_val = (val_A < 0.5).squeeze()
                    if mask0_val.sum() > 0:
                        input0 = torch.cat((val_Z[mask0_val], val_X[mask0_val]), dim=1)
                        p0 = model0(input0)
                        loss0_val = self.q_loss(p0, mask0_val, k_val)
                    else:
                        loss0_val = 0.0

                    mask1_val = (val_A > 0.5).squeeze()
                    if mask1_val.sum() > 0:
                        input1 = torch.cat((val_Z[mask1_val], val_X[mask1_val]), dim=1)
                        p1 = model1(input1)
                        loss1_val = self.q_loss(p1, mask1_val, k_val)
                    else:
                        loss1_val = 0.0
                    causal_loss_val_full = loss0_val + loss1_val   
                    
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val_full, epoch)
                    self.causal_val_losses.append(causal_loss_val_full.item()) # .item()
                
                scheduler0.step(causal_loss_val_full.item())
                scheduler1.step(causal_loss_val_full.item())
                early_stopping(causal_loss_val_full.item(), model0, model1)
                
                if early_stopping.early_stop:
                    model0.load_state_dict(early_stopping.best_model0_state)
                    model1.load_state_dict(early_stopping.best_model1_state)
                    if verbose:
                        print(f"\n[Early Stopping] 在 Epoch {epoch} 停止训练。")
                        print(f"最佳验证集 Loss: {early_stopping.best_loss:.6f}")
                    break # 跳出 epoch 循环
        #print("训练完成。")
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

        
        with torch.no_grad():

            q_hat = model(Z_samples, X_samples, A_samples)

        signs = torch.where(A_samples > 0.5, torch.ones_like(A_samples), -torch.ones_like(A_samples))

        phi_hat = (signs * q_hat * Y_samples).mean()

        return phi_hat.cpu()
    
    
    
class NMMR_Q_Trainer_SGD(NMMR_Q_Trainer):
    pass
        
class NMMR_Q_Trainer_RHC(NMMR_Q_Trainer):

    pass