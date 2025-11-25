import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from src.models.NMMRh.nmmr_h_loss import NMMR_H_Loss
from src.models.NMMRh.nmmr_h_model import MLP_for_NMMR

from src.models.NMMRh.kernel_h_utils import fit_sigma, G_kernel, calculate_kernel_matrix_batched

def _resolve_dataset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset, dataset.indices
    return dataset, None

def _select_rows(tensor, indices):
    if indices is None:
        return tensor
    return tensor[indices]

class NMMR_H_Trainer:
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
        
        # Kernel Gamma
        self.kernel_gamma = train_params.get('kernel_gamma', 1.0)

        self.scaler = GradScaler() if self.gpu_flg else None
        self.device = torch.device('cuda' if self.gpu_flg else 'cpu')
        
        # 初始化 H-Loss
        self.h_loss = NMMR_H_Loss(
            use_u_statistic=self.train_params.get('use_u_statistic', False)
        )

        self.writer = None
        if self.log_metrics and dump_folder is not None:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []
        else:
            self.log_metrics = False

    def _compute_kernel_matrix(self, batch_A, batch_Z, batch_X):
        """
        NMMR (h) 的核定义在工具变量空间 V = (A, Z, X) 上
        Kernel Input: Concat(A, Z, X)
        """
        kernel_inputs = torch.cat((batch_A, batch_Z, batch_X), dim=1)
        N = kernel_inputs.shape[0]
        
        # 动态计算 Sigma (与 q-trainer 一致)
        sigma_data = fit_sigma(kernel_inputs)
        
        # 计算核矩阵
        k_matrix = calculate_kernel_matrix_batched(
            dataset=kernel_inputs,
            batch_indices=(0, N),
            kernel=G_kernel,
            sigma=sigma_data,
            gamma=self.kernel_gamma,
        )
        return k_matrix

    def train(self, train_loader: DataLoader, val_loader: DataLoader, verbose: int = 1) -> MLP_for_NMMR:
        
        train_dataset, train_indices = _resolve_dataset(train_loader.dataset)
        val_dataset, val_indices = _resolve_dataset(val_loader.dataset)

        A_dim = train_dataset.A.shape[1]
        W_dim = train_dataset.W.shape[1]
        X_dim = train_dataset.X.shape[1]
        input_size = A_dim + W_dim + X_dim
        
        model = MLP_for_NMMR(input_dim=input_size, train_params=self.train_params)
        
        if self.gpu_flg:
            model.cuda()

        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        for epoch in tqdm(range(self.n_epochs), desc="Epochs", disable=not verbose):
            model.train()
            
            for batch_data in train_loader:

                batch_A = batch_data['A'].to(self.device)
                batch_W = batch_data['W'].to(self.device)
                batch_Z = batch_data['Z'].to(self.device)
                batch_X = batch_data['X'].to(self.device)
                batch_Y = batch_data['Y'].to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    k_matrix = self._compute_kernel_matrix(batch_A, batch_Z, batch_X)

                with autocast(enabled=self.gpu_flg):

                    model_input = torch.cat((batch_A, batch_W, batch_X), dim=1)
                    pred_y = model(model_input)

                    loss = self.h_loss(model_output=pred_y, target=batch_Y, kernel_matrix=k_matrix)

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # --- Logging ---
            if self.log_metrics:
                model.eval()
                with torch.no_grad():
                    # --- 训练集评估 ---
                    t_A = _select_rows(train_dataset.A, train_indices).to(self.device)
                    t_W = _select_rows(train_dataset.W, train_indices).to(self.device)
                    t_Z = _select_rows(train_dataset.Z, train_indices).to(self.device)
                    t_X = _select_rows(train_dataset.X, train_indices).to(self.device)
                    t_Y = _select_rows(train_dataset.Y, train_indices).to(self.device)

                    t_in = torch.cat((t_A, t_W, t_X), dim=1)
                    t_pred = model(t_in)
                    t_k = self._compute_kernel_matrix(t_A, t_Z, t_X)
                    train_loss = self.h_loss(t_pred, t_Y, t_k)
                    
                    self.writer.add_scalar(f'{self.loss_name}/train', train_loss, epoch)
                    self.causal_train_losses.append(train_loss.item() if torch.is_tensor(train_loss) else train_loss)

                    # --- 验证集评估 ---
                    v_A = _select_rows(val_dataset.A, val_indices).to(self.device)
                    v_W = _select_rows(val_dataset.W, val_indices).to(self.device)
                    v_Z = _select_rows(val_dataset.Z, val_indices).to(self.device)
                    v_X = _select_rows(val_dataset.X, val_indices).to(self.device)
                    v_Y = _select_rows(val_dataset.Y, val_indices).to(self.device)

                    v_in = torch.cat((v_A, v_W, v_X), dim=1)
                    v_pred = model(v_in)
                    v_k = self._compute_kernel_matrix(v_A, v_Z, v_X)
                    val_loss = self.h_loss(v_pred, v_Y, v_k)

                    self.writer.add_scalar(f'{self.loss_name}/val', val_loss, epoch)
                    self.causal_val_losses.append(val_loss.item() if torch.is_tensor(val_loss) else val_loss)

        return model

    @staticmethod
    def predict(model: MLP_for_NMMR, dataset_view):
        """
        ATE = E[Y(1) - Y(0)] = E[h(1, W, X) - h(0, W, X)]
        """
        model.eval()
        model_device = next(model.parameters()).device

        W = dataset_view.W.to(model_device)
        X = dataset_view.X.to(model_device)
        N = W.shape[0]

        ones = torch.ones((N, 1), device=model_device)
        zeros = torch.zeros((N, 1), device=model_device)
        
        # h(1, W, X)
        input_1 = torch.cat((ones, W, X), dim=1)
        # h(0, W, X)
        input_0 = torch.cat((zeros, W, X), dim=1)
        
        with torch.no_grad():
            h_1 = model(input_1)
            h_0 = model(input_0)
            
        # ATE = mean(h_1 - h_0)
        ate = (h_1 - h_0).mean()
        
        return ate.cpu()
    
class NMMR_H_Trainer_SGD(NMMR_H_Trainer):
    pass

class NMMR_H_Trainer_RHC(NMMR_H_Trainer):
    pass