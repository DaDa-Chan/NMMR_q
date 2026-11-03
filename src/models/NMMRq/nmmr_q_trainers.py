import os.path as op
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import copy
import datetime
import math
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


from src.models.NMMRq.nmmr_q_loss import NMMR_Q_Loss
from src.models.NMMRq.nmmr_q_model import NMMR_Q_common, NMMR_Q_DualModel
from src.models.kernel_utils import fit_sigma, G_kernel, calculate_kernel_matrix_batched
from src.models.experiment_common import to_bool, check_loss_name_matches_u_statistic, TrainResult
from src.utils import get_device


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
    Tracks the best validation score seen so far (and the corresponding
    model checkpoints), and flags when `patience` epochs have passed
    without improvement. Callers decide whether to act on `early_stop`;
    the best checkpoint is tracked unconditionally whenever this is called
    on a finite score, independent of whether early stopping is enabled.
    """
    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience (int): epochs to wait after the last improvement before flagging early_stop.
            min_delta (float): minimum change to count as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model0_state = None
        self.best_model1_state = None
        self.best_epoch = None

    def __call__(self, val_loss, model0, model1, epoch):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model0, model1)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model0, model1):
        """Save the current best model parameters in memory."""
        self.best_model0_state = copy.deepcopy(model0.state_dict())
        self.best_model1_state = copy.deepcopy(model1.state_dict())

class NMMR_Q_Trainer:
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], random_seed: int,
                 dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.train_params = train_params
        self.random_seed = random_seed
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.device = get_device()
        self.use_amp = self.device.type == 'cuda'
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        # Multiplier applied to the fitted median-heuristic bandwidth; defaults to 1.0
        # (i.e. use the fitted bandwidth as-is) so existing configs are unaffected.
        self.kernel_bandwidth_multiplier = float(train_params.get('kernel_bandwidth_multiplier', 1.0))
        self._frozen_sigma: Optional[float] = None

        check_loss_name_matches_u_statistic(train_params)
        self.q_loss = NMMR_Q_Loss(
            use_u_statistic=self.train_params.get('use_u_statistic', False),
        )

        # These four switches are independent: logging must never change what
        # training actually does. A caller with no val_loader (e.g. a final
        # refit with no held-out data) should set all three validation-driven
        # switches to False explicitly, rather than relying on log_metrics.
        self.use_validation = to_bool(train_params.get('use_validation', True))
        self.use_early_stopping = to_bool(train_params.get('use_early_stopping', True))
        self.use_scheduler = to_bool(train_params.get('use_scheduler', True))
        self.log_metrics = to_bool(train_params.get('log_metrics', False))

        if (self.use_early_stopping or self.use_scheduler) and not self.use_validation:
            raise ValueError(
                "use_early_stopping/use_scheduler require use_validation=True "
                "(both need a validation score to act on)."
            )

        self.writer = None
        if self.log_metrics and dump_folder is not None:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def _compute_kernel_matrix(self, w, x):
        """
        Compute the full-batch kernel matrix.
        Inputs: w [N, w_dim], x [N, x_dim]. Output: K [N, N].
        Uses the bandwidth frozen once per fold by `_freeze_kernel_bandwidth`
        (set at the start of `train`) instead of refitting it on whatever
        batch or split happens to be passed in, which would let the
        validation/eval split leak into the kernel and make the effective
        kernel vary from batch to batch.
        """
        N = w.shape[0]
        wx_group = torch.cat([w, x], dim=1)  # [N, D]
        if self._frozen_sigma is None:
            raise RuntimeError(
                "Kernel bandwidth has not been frozen yet; call _freeze_kernel_bandwidth "
                "on the training subset before computing any kernel matrix."
            )
        k_matrix = calculate_kernel_matrix_batched(
            dataset=wx_group,
            batch_indices=(0, N),
            kernel=G_kernel,
            sigma=self._frozen_sigma,
            gamma=None,
        )
        return k_matrix

    def _freeze_kernel_bandwidth(self, w, x) -> float:
        """
        Fit the median-heuristic bandwidth once from the given (training-only)
        subset and freeze it for the rest of this fold's training, validation,
        and HPO scoring. Must be called with training data only.
        """
        wx_group = torch.cat([w, x], dim=1)
        base_sigma = fit_sigma(wx_group)
        self._frozen_sigma = base_sigma * self.kernel_bandwidth_multiplier
        return self._frozen_sigma
    
    def _q_loss_on_split(self, model0, model1, A, W, Z, X, kernel_matrix):
        """Compute the combined q-loss for one split, given a pre-built kernel matrix."""
        mask0 = (A < 0.5).squeeze()
        if mask0.sum() > 0:
            input0 = torch.cat((Z[mask0], X[mask0]), dim=1)
            p0 = model0(input0)
            loss0 = self.q_loss(p0, mask0, kernel_matrix)
        else:
            loss0 = torch.zeros((), device=self.device)

        mask1 = (A > 0.5).squeeze()
        if mask1.sum() > 0:
            input1 = torch.cat((Z[mask1], X[mask1]), dim=1)
            p1 = model1(input1)
            loss1 = self.q_loss(p1, mask1, kernel_matrix)
        else:
            loss1 = torch.zeros((), device=self.device)

        return loss0 + loss1

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              verbose: int = 0) -> Tuple[NMMR_Q_DualModel, TrainResult]:
        """
        Train the NMMR_Q model: q0(Z, X) & q1(Z, X).

        `use_validation`/`use_early_stopping`/`use_scheduler`/`log_metrics` are
        independent switches (see __init__): disabling logging never changes
        training behavior, and validation/early-stopping/scheduling run
        whenever their switch is on, regardless of whether a dump_folder was
        given for TensorBoard output.
        """
        start_time = time.monotonic()

        train_dataset, train_indices = _resolve_dataset(train_loader.dataset)
        if self.use_validation:
            if val_loader is None:
                raise ValueError("use_validation=True but no val_loader was provided.")
            val_dataset, val_indices = _resolve_dataset(val_loader.dataset)

        Z_dim = train_dataset.Z.shape[1]
        X_dim = train_dataset.X.shape[1]
        input_size = Z_dim + X_dim

        model0 = NMMR_Q_common(input_dim=input_size, train_params=self.train_params)
        model1 = NMMR_Q_common(input_dim=input_size, train_params=self.train_params)

        model0.to(self.device)
        model1.to(self.device)

        optimizer0 = optim.Adam(list(model0.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)
        optimizer1 = optim.Adam(list(model1.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        # Freeze the kernel bandwidth once, from the training subset only, so it
        # cannot drift batch-to-batch and cannot be fit on validation data.
        train_W_all = _select_rows(train_dataset.W, train_indices).to(self.device)
        train_X_all = _select_rows(train_dataset.X, train_indices).to(self.device)
        self._freeze_kernel_bandwidth(train_W_all, train_X_all)

        scheduler0 = scheduler1 = None
        if self.use_scheduler:
            sched_patience = self.train_params.get('scheduler_patience', 5)
            sched_factor = self.train_params.get('scheduler_factor', 0.5)
            sched_min_lr = self.train_params.get('scheduler_min_lr', 1e-6)
            scheduler0 = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer0, mode='min', factor=sched_factor, patience=sched_patience, min_lr=sched_min_lr
            )
            scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer1, mode='min', factor=sched_factor, patience=sched_patience, min_lr=sched_min_lr
            )

        early_stopping = None
        if self.use_validation:
            patience = self.train_params.get('patience', 20)
            min_delta = self.train_params.get('early_stopping_min_delta', 5e-5)
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        n_nonfinite_train = 0
        n_nonfinite_val = 0
        n_lr_reductions = 0
        stop_reason = "max_epochs"
        stopped_early = False
        last_epoch = -1

        for epoch in tqdm(range(self.n_epochs), desc="Epochs", disable=not verbose):
            last_epoch = epoch
            model0.train()
            model1.train()

            for batch_data in train_loader:
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
                    with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
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
                    with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
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

            if not self.use_validation:
                continue

            # --- Validation (independent of whether we log it) ---
            model0.eval()
            model1.eval()
            with torch.no_grad():
                train_A = _select_rows(train_dataset.A, train_indices).to(self.device)
                train_Z = _select_rows(train_dataset.Z, train_indices).to(self.device)
                k_train = self._compute_kernel_matrix(train_W_all, train_X_all)
                total_train_loss = self._q_loss_on_split(
                    model0, model1, train_A, train_W_all, train_Z, train_X_all, k_train
                )
                if not math.isfinite(total_train_loss.item()):
                    n_nonfinite_train += 1

                val_A = _select_rows(val_dataset.A, val_indices).to(self.device)
                val_W = _select_rows(val_dataset.W, val_indices).to(self.device)
                val_Z = _select_rows(val_dataset.Z, val_indices).to(self.device)
                val_X = _select_rows(val_dataset.X, val_indices).to(self.device)
                k_val = self._compute_kernel_matrix(val_W, val_X)
                causal_loss_val_full = self._q_loss_on_split(
                    model0, model1, val_A, val_W, val_Z, val_X, k_val
                )

            val_score = causal_loss_val_full.item()
            val_is_finite = math.isfinite(val_score)
            if not val_is_finite:
                n_nonfinite_val += 1

            if self.log_metrics and self.writer is not None:
                self.writer.add_scalar(f'{self.loss_name}/train', total_train_loss, epoch)
                self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val_full, epoch)
                self.causal_train_losses.append(total_train_loss.item())
                self.causal_val_losses.append(val_score)

            if val_is_finite and self.use_scheduler:
                lr_before = optimizer0.param_groups[0]['lr']
                scheduler0.step(val_score)
                scheduler1.step(val_score)
                if optimizer0.param_groups[0]['lr'] < lr_before:
                    n_lr_reductions += 1

            if val_is_finite:
                # Best-checkpoint tracking always runs on a finite score, whether
                # or not we act on `early_stop` below.
                early_stopping(val_score, model0, model1, epoch)

            if self.use_early_stopping and early_stopping.early_stop:
                stop_reason = "early_stopping"
                stopped_early = True
                if verbose:
                    print(f"\n[Early Stopping] stopped at epoch {epoch}.")
                    print(f"Best validation loss: {early_stopping.best_loss:.6f}")
                break

        train_seconds = time.monotonic() - start_time
        best_state_restored = False
        best_epoch = 0
        best_val_score = None

        if self.use_validation:
            if early_stopping.best_loss is not None:
                # Unconditionally restore the best checkpoint, whether we exited via
                # early stopping or simply ran out of epochs.
                model0.load_state_dict(early_stopping.best_model0_state)
                model1.load_state_dict(early_stopping.best_model1_state)
                best_state_restored = True
                best_epoch = early_stopping.best_epoch
                best_val_score = early_stopping.best_loss
            else:
                stop_reason = "no_valid_checkpoint"
                print("[Warning] validation never produced a finite score for this fold; "
                      "returning last-epoch weights instead of a selected-best checkpoint.")

        result = TrainResult(
            best_epoch=best_epoch,
            best_val_score=best_val_score,
            last_epoch=last_epoch,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            n_lr_reductions=n_lr_reductions,
            final_learning_rate=optimizer0.param_groups[0]['lr'],
            train_seconds=train_seconds,
            best_state_restored=best_state_restored,
            n_nonfinite_train=n_nonfinite_train,
            n_nonfinite_val=n_nonfinite_val,
            kernel_bandwidth=self._frozen_sigma,
            seed=self.random_seed,
        )
        return NMMR_Q_DualModel(model0, model1), result

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
        
        # --- DEBUG: 检查 q 值和 Y 值分布（取消注释以启用）---
        mask1 = (A_samples > 0.5).squeeze()
        mask0 = ~mask1
        # print(f"  [Predict] n={A_samples.shape[0]}, n1={mask1.sum().item()}, n0={mask0.sum().item()}")
        # print(f"  [Predict] Y: mean={Y_samples.mean():.4f}, min={Y_samples.min():.4f}, max={Y_samples.max():.4f}")
        # print(f"  [Predict] q(A=1): mean={q_hat[mask1].mean():.4f}, std={q_hat[mask1].std():.4f}, min={q_hat[mask1].min():.4f}, max={q_hat[mask1].max():.4f}")
        # print(f"  [Predict] q(A=0): mean={q_hat[mask0].mean():.4f}, std={q_hat[mask0].std():.4f}, min={q_hat[mask0].min():.4f}, max={q_hat[mask0].max():.4f}")
        # print(f"  [Predict] sign*q*Y: pos_sum={( signs * q_hat * Y_samples)[mask1].sum():.4f}, neg_sum={(signs * q_hat * Y_samples)[mask0].sum():.4f}")
        
        phi_hat = (signs * q_hat * Y_samples).mean()

        return phi_hat.cpu()
    
    
    
class NMMR_Q_Trainer_SGD(NMMR_Q_Trainer):
    pass
        
class NMMR_Q_Trainer_RHC(NMMR_Q_Trainer):

    pass