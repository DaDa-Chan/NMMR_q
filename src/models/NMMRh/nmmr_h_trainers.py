import os.path as op
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import copy
import math
import time

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from src.models.NMMRh.nmmr_h_loss import NMMR_H_Loss
from src.models.NMMRh.nmmr_h_model import MLP_for_NMMR
from src.models.kernel_utils import fit_sigma, G_kernel, calculate_kernel_matrix_batched
from src.models.experiment_common import to_bool, check_loss_name_matches_u_statistic, TrainResult
from src.utils import get_device


def _resolve_dataset(dataset):
    if isinstance(dataset, Subset):
        return (dataset.dataset, dataset.indices)
    else:
        return (dataset, None)


def _select_rows(tensor, indices):
    if indices is None:
        return tensor
    else:
        return tensor[indices]


class EarlyStopping:
    """
    Tracks the best validation score seen so far (and the corresponding
    model checkpoint), and flags when `patience` epochs have passed without
    improvement. The best checkpoint is tracked unconditionally whenever
    this is called on a finite score, independent of whether early stopping
    is enabled by the caller.
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
        self.best_model_state = None
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Save the current best model parameters in memory."""
        self.best_model_state = copy.deepcopy(model.state_dict())


class NMMR_H_Trainer:
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], random_seed: int, dump_folder: Optional[Path] = None):
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
        # Multiplier applied to the fitted median-heuristic bandwidth; defaults to 1.0
        # (i.e. use the fitted bandwidth as-is) so existing configs are unaffected.
        self.kernel_bandwidth_multiplier = float(train_params.get('kernel_bandwidth_multiplier', 1.0))
        self._frozen_sigma: Optional[float] = None
        # Fixed scoring bandwidth: median heuristic x 1, independent of the candidate's
        # kernel_bandwidth_multiplier, so held-out scores are comparable across candidates.
        self._reference_sigma: Optional[float] = None
        # PyTorch 2.2 exposes GradScaler under torch.cuda.amp; the
        # torch.amp.GradScaler entry point was added in a later release.
        self.scaler = GradScaler() if self.use_amp else None

        check_loss_name_matches_u_statistic(train_params)
        self.h_loss = NMMR_H_Loss(use_u_statistic=self.train_params.get('use_u_statistic', False))

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
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f'tensorboard_log_{random_seed}'))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def _compute_kernel_matrix(self, batch_A, batch_Z, batch_X):
        """
        NMMR (h)'s kernel is defined on the instrument space V = (A, Z, X).
        Uses the bandwidth frozen once per fold by `_freeze_kernel_bandwidth`
        (set at the start of `train`) instead of refitting it on whatever
        batch or split happens to be passed in, which would let the
        validation/eval split leak into the kernel and make the effective
        kernel vary from batch to batch.
        """
        kernel_inputs = torch.cat((batch_A, batch_Z, batch_X), dim=1)
        N = kernel_inputs.shape[0]
        if self._frozen_sigma is None:
            raise RuntimeError(
                "Kernel bandwidth has not been frozen yet; call _freeze_kernel_bandwidth "
                "on the training subset before computing any kernel matrix."
            )
        k_matrix = calculate_kernel_matrix_batched(
            dataset=kernel_inputs, batch_indices=(0, N), kernel=G_kernel,
            sigma=self._frozen_sigma, gamma=None,
        )
        return k_matrix

    def _compute_reference_kernel_matrix(self, batch_A, batch_Z, batch_X):
        """
        Scoring kernel on the same (A, Z, X) inputs, but with the fixed
        reference bandwidth (median heuristic x 1) instead of the candidate's
        training bandwidth. Used by every held-out moment-risk score.
        """
        if self._reference_sigma is None:
            raise RuntimeError(
                "Reference bandwidth has not been set yet; call _freeze_kernel_bandwidth "
                "on the training subset before computing any scoring kernel."
            )
        kernel_inputs = torch.cat((batch_A, batch_Z, batch_X), dim=1)
        return calculate_kernel_matrix_batched(
            dataset=kernel_inputs, batch_indices=(0, kernel_inputs.shape[0]), kernel=G_kernel,
            sigma=self._reference_sigma, gamma=None,
        )

    def _freeze_kernel_bandwidth(self, A, Z, X) -> float:
        """
        Fit the median-heuristic bandwidth once from the given (training-only)
        subset. The training kernel (`_frozen_sigma` = median x the candidate's
        multiplier) is frozen for this fold's training and early-stopping
        validation; the scoring kernel (`_reference_sigma` = median x 1) is
        frozen for held-out scoring. Must be called with training data only.
        """
        kernel_inputs = torch.cat((A, Z, X), dim=1)
        base_sigma = fit_sigma(kernel_inputs)
        self._reference_sigma = base_sigma
        self._frozen_sigma = base_sigma * self.kernel_bandwidth_multiplier
        return self._frozen_sigma

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              verbose: int = 0) -> Tuple[MLP_for_NMMR, TrainResult]:
        """
        Train the NMMR_H model.

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

        A_dim = train_dataset.A.shape[1]
        W_dim = train_dataset.W.shape[1]
        X_dim = train_dataset.X.shape[1]
        input_size = A_dim + W_dim + X_dim
        model = MLP_for_NMMR(input_dim=input_size, train_params=self.train_params)
        model.to(self.device)
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        # Freeze the kernel bandwidth once, from the training subset only, so it
        # cannot drift batch-to-batch and cannot be fit on validation data.
        train_A_all = _select_rows(train_dataset.A, train_indices).to(self.device)
        train_W_all = _select_rows(train_dataset.W, train_indices).to(self.device)
        train_Z_all = _select_rows(train_dataset.Z, train_indices).to(self.device)
        train_X_all = _select_rows(train_dataset.X, train_indices).to(self.device)
        train_Y_all = _select_rows(train_dataset.Y, train_indices).to(self.device)
        self._freeze_kernel_bandwidth(train_A_all, train_Z_all, train_X_all)

        scheduler = None
        if self.use_scheduler:
            sched_patience = self.train_params.get('scheduler_patience', 5)
            sched_factor = self.train_params.get('scheduler_factor', 0.5)
            sched_min_lr = self.train_params.get('scheduler_min_lr', 1e-6)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=sched_factor, patience=sched_patience, min_lr=sched_min_lr
            )

        early_stopping = None
        if self.use_validation:
            patience = self.train_params.get('patience', 20)
            min_delta = self.train_params.get('early_stopping_min_delta', 1e-05)
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        n_nonfinite_train = 0
        n_nonfinite_val = 0
        n_lr_reductions = 0
        stop_reason = "max_epochs"
        stopped_early = False
        last_epoch = -1

        for epoch in tqdm(range(self.n_epochs), desc='Epochs', disable=not verbose):
            last_epoch = epoch
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
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
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

            if not self.use_validation:
                continue

            # --- Validation (independent of whether we log it) ---
            model.eval()
            with torch.no_grad():
                t_in = torch.cat((train_A_all, train_W_all, train_X_all), dim=1)
                t_pred = model(t_in)
                t_k = self._compute_kernel_matrix(train_A_all, train_Z_all, train_X_all)
                train_loss = self.h_loss(t_pred, train_Y_all, t_k)
                if not math.isfinite(train_loss.item()):
                    n_nonfinite_train += 1

                v_A = _select_rows(val_dataset.A, val_indices).to(self.device)
                v_W = _select_rows(val_dataset.W, val_indices).to(self.device)
                v_Z = _select_rows(val_dataset.Z, val_indices).to(self.device)
                v_X = _select_rows(val_dataset.X, val_indices).to(self.device)
                v_Y = _select_rows(val_dataset.Y, val_indices).to(self.device)
                v_in = torch.cat((v_A, v_W, v_X), dim=1)
                v_pred = model(v_in)
                v_k = self._compute_kernel_matrix(v_A, v_Z, v_X)
                val_loss = self.h_loss(v_pred, v_Y, v_k)

            val_score = val_loss.item()
            val_is_finite = math.isfinite(val_score)
            if not val_is_finite:
                n_nonfinite_val += 1

            if self.log_metrics and self.writer is not None:
                self.writer.add_scalar(f'{self.loss_name}/train', train_loss, epoch)
                self.writer.add_scalar(f'{self.loss_name}/val', val_loss, epoch)
                self.causal_train_losses.append(train_loss.item())
                self.causal_val_losses.append(val_score)

            if val_is_finite and self.use_scheduler:
                lr_before = optimizer.param_groups[0]['lr']
                scheduler.step(val_score)
                if optimizer.param_groups[0]['lr'] < lr_before:
                    n_lr_reductions += 1

            if val_is_finite:
                # Best-checkpoint tracking always runs on a finite score, whether
                # or not we act on `early_stop` below.
                early_stopping(val_score, model, epoch)

            if self.use_early_stopping and early_stopping.early_stop:
                stop_reason = "early_stopping"
                stopped_early = True
                if verbose:
                    print(f'\n[Early Stopping] stopped at epoch {epoch}.')
                    print(f'Best validation loss: {early_stopping.best_loss:.6f}')
                break

        train_seconds = time.monotonic() - start_time
        best_state_restored = False
        best_epoch = 0
        best_val_score = None

        if self.use_validation:
            if early_stopping.best_loss is not None:
                # Unconditionally restore the best checkpoint, whether we exited via
                # early stopping or simply ran out of epochs.
                model.load_state_dict(early_stopping.best_model_state)
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
            final_learning_rate=optimizer.param_groups[0]['lr'],
            train_seconds=train_seconds,
            best_state_restored=best_state_restored,
            n_nonfinite_train=n_nonfinite_train,
            n_nonfinite_val=n_nonfinite_val,
            kernel_bandwidth=self._frozen_sigma,
            seed=self.random_seed,
        )
        return model, result

    @staticmethod
    def predict(model: MLP_for_NMMR, dataset_view):
        """
        NMMR (h) ATE prediction: ATE = E[Y(1) - Y(0)] = E[h(1, W, X) - h(0, W, X)]
        """
        model.eval()
        model_device = next(model.parameters()).device
        W = dataset_view.W.to(model_device)
        X = dataset_view.X.to(model_device)
        N = W.shape[0]
        ones = torch.ones((N, 1), device=model_device)
        zeros = torch.zeros((N, 1), device=model_device)
        input_1 = torch.cat((ones, W, X), dim=1)
        input_0 = torch.cat((zeros, W, X), dim=1)
        with torch.no_grad():
            h_1 = model(input_1)
            h_0 = model(input_0)
        ate = (h_1 - h_0).mean()
        return ate.cpu()


class NMMR_H_Trainer_SGD(NMMR_H_Trainer):
    pass


class NMMR_H_Trainer_RHC(NMMR_H_Trainer):
    pass
