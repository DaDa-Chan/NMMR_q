import torch
import numpy as np
import pandas as pd
import copy
import math
import optuna

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.models.NMMRq.nmmr_q_trainers import NMMR_Q_Trainer,NMMR_Q_Trainer_SGD, NMMR_Q_Trainer_RHC, NMMR_Q_DualModel
from src.models.NMMRq.nmmr_q_model import NMMR_Q_common
from src.models.NMMRq.nmmr_q_loss import NMMR_Q_Loss
from src.data.ate.sgd_pv import generate_data
from src.data.ate.data_class import SGDDataset, RHCDataset, MergedDataset
from src.models.experiment_common import (
    _locate_split_file,
    _parse_optuna_space,
    _suggest_from_trial,
    _build_loaders_for_fold,
    _build_dataset_view,
    nested_hpo_fold_split,
    weighted_fold_mean,
    fold_score_dispersion,
    insufficient_treatment_groups_reason,
    select_one_se_candidate,
    fold_control_seeds,
    build_fold_record,
    finite_valued_trials,
    summarize_trial_failure_reasons,
    NoFiniteRound1TrialError,
)
from src.utils import get_device, empty_device_cache, set_random_seed, make_seed_bundle

# Pre-specified, fixed-ahead-of-time failure thresholds for the pilot HPO
# selection flow (never adjusted post-hoc based on a particular run's
# results). Overridable per method via the `q_hard_threshold` /
# `max_fold_train_seconds` keys in that method's config.
DEFAULT_Q_HARD_THRESHOLD = 1000.0
DEFAULT_MAX_FOLD_TRAIN_SECONDS = 1800.0


def _create_trainer(data_configs, train_params, random_seed, dump_folder, data_name):
    
    if data_name == 'sgd':      
        return NMMR_Q_Trainer_SGD(
            data_configs=data_configs,
            train_params=train_params,
            random_seed=random_seed,
            dump_folder=dump_folder,
        )
    elif data_name == 'rhc' :
        return NMMR_Q_Trainer_RHC(
            data_configs=data_configs,
            train_params=train_params,
            random_seed=random_seed,
            dump_folder=dump_folder,
        )

def _compute_unified_q_score(trainer,
                             model: torch.nn.Module,
                             dataset,
                             indices: List[int]) -> Tuple[float, int]:
    """
    Unified HPO scoring metric for Q (plan section 5.4): regardless of
    whether this candidate trained with the U- or V-statistic objective,
    candidates are always compared on the V-statistic moment risk
        S_Q(lambda) = 0.5 * sum_{a in {0,1}} (1/n^2) rho_a^T K rho_a,
    evaluated on `indices` -- a held-out "score" fold that must never have
    participated in this candidate's gradient training or early stopping
    (see nested_hpo_fold_split). Uses the kernel bandwidth frozen by the
    trainer on its own gradient-training subset, not refit here.
    """
    view = _build_dataset_view(dataset, indices)
    scorer = NMMR_Q_Loss(use_u_statistic=False)
    with torch.no_grad():
        A = view.A.to(trainer.device)
        W = view.W.to(trainer.device)
        X = view.X.to(trainer.device)
        Z = view.Z.to(trainer.device)

        k_matrix = trainer._compute_kernel_matrix(W, X)
        score_total = torch.zeros((), device=trainer.device)

        mask0 = (A < 0.5).squeeze()
        if mask0.sum() > 0:
            inputs0 = torch.cat((Z[mask0], X[mask0]), dim=1)
            pred0 = model.net0(inputs0)
            score_total = score_total + scorer(pred0, mask0, k_matrix)

        mask1 = (A > 0.5).squeeze()
        if mask1.sum() > 0:
            inputs1 = torch.cat((Z[mask1], X[mask1]), dim=1)
            pred1 = model.net1(inputs1)
            score_total = score_total + scorer(pred1, mask1, k_matrix)

    unified_score = 0.5 * score_total
    return float(unified_score.item()), len(indices)


def _q_predictions_by_treatment(trainer, model, dataset, indices) -> Dict[int, np.ndarray]:
    """
    Raw q predictions on `indices`, split by the treatment level each
    sample actually received (q0(Z,X) for A=0 rows, q1(Z,X) for A=1 rows).
    Used for both the hard-threshold failure check (pilot HPO fix request
    item 3) and the Round 1/Round 2 record's per-treatment-group
    distribution stats (item 7) -- one forward pass serves both.
    """
    view = _build_dataset_view(dataset, indices)
    with torch.no_grad():
        A = view.A.to(trainer.device)
        Z = view.Z.to(trainer.device)
        X = view.X.to(trainer.device)
        inputs = torch.cat((Z, X), dim=1)
        q0 = model.net0(inputs).detach().cpu().numpy().reshape(-1)
        q1 = model.net1(inputs).detach().cpu().numpy().reshape(-1)
        a = A.detach().cpu().numpy().reshape(-1)
    return {0: q0[a < 0.5], 1: q1[a > 0.5]}


def _max_abs_q(q_by_treatment: Dict[int, np.ndarray]) -> float:
    values = [arr for arr in q_by_treatment.values() if arr.size > 0]
    if not values:
        return 0.0
    return float(max(np.abs(arr).max() for arr in values))


def _q_distribution_stats(q_by_treatment: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Per-treatment-group Q distribution diagnostics (pilot HPO fix request
    item 7): mean/SD/min/max, P50/P90/P95/P99, and q>10/50/100 proportions."""
    stats: Dict[str, float] = {}
    for level, q_level in q_by_treatment.items():
        if q_level.size == 0:
            continue
        stats[f"q_a{level}_mean"] = float(q_level.mean())
        stats[f"q_a{level}_sd"] = float(q_level.std())
        stats[f"q_a{level}_min"] = float(q_level.min())
        stats[f"q_a{level}_max"] = float(q_level.max())
        for pct in (50, 90, 95, 99):
            stats[f"q_a{level}_p{pct}"] = float(np.percentile(q_level, pct))
        stats[f"q_a{level}_prop_gt10"] = float((q_level > 10).mean())
        stats[f"q_a{level}_prop_gt50"] = float((q_level > 50).mean())
        stats[f"q_a{level}_prop_gt100"] = float((q_level > 100).mean())
    return stats


def _run_optuna_tuning(
    data_name: str,
    dataset,
    n_splits: int,
    n_trials: int,
    base_train_params: Dict[str, Any],
    search_space: Dict[str, Any],
    data_configs: Dict[str, Any],
    random_seed: int,
    log_folder:Optional[Path] = None,
    save_dir: Optional[Path] = None,
    scenario: Optional[int] = 1,
    tuning_seed: Optional[int] = None,
    return_study: bool = False,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    existing_trial_count: int = 0,
):
    """
    1.在给定数据集上执行 K-fold CV 以进行 Optuna 调参。返回最优的超参数组合。
    2.如果提供了 save_dir, 则使用最佳参数在全量 dataset 上重新训练并保存模型。

    Returns `final_best_params` (dict) by default. If `return_study=True`,
    returns `(final_best_params, study, selected_trial, best_trial)` instead,
    so a caller (e.g. hpo_protocol.py) can archive the full trial table and
    the one-SE vs pure-minimum provenance (plan section 6.2).

    `storage`/`study_name`: when given (e.g. "sqlite:///tune_results/
    optuna_studies/q_u_round1.db"), the Optuna study is persisted there
    (`load_if_exists=True`) instead of living only in memory, so a run
    interrupted partway through can resume and add trials rather than
    starting over (pilot HPO fix request item 7).

    `existing_trial_count`: how many trials the persisted study already
    has BEFORE this call (0 for a brand-new study). A resumed run must NOT
    reuse the exact same `tuning_seed` to build its TPESampler -- the
    sampler itself is never persisted, only the trials are, so a freshly
    constructed `TPESampler(seed=tuning_seed)` would otherwise replay the
    EXACT SAME "random startup" draws every time, silently re-suggesting
    the whole first batch of hyperparameters again on every resume
    (精简与config管理.md Phase A follow-up: "恢复运行可能重复抽到相同参数").
    When `existing_trial_count > 0`, the sampler seed is deterministically
    re-derived from (tuning_seed, existing_trial_count) instead, so a
    resumed batch is not forced to replay the first batch wholesale. This
    does not guarantee zero overlap with any earlier trial -- a different
    sampler seed can still land on the same point by chance (e.g. a
    categorical choice with few options) -- only that resuming no longer
    *systematically* reproduces the previous batch via an unchanged
    sampler seed.
    """
    if not search_space:
        print("[Optuna] 未提供搜索空间，跳过调优。")
        if return_study:
            return base_train_params, None, None, None
        return base_train_params

    # 减少 Optuna 日志
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        # 为当前试验生成参数
        current_params = copy.deepcopy(base_train_params)
        for param_name, config in search_space.items():
            current_params[param_name] = _suggest_from_trial(
                trial, param_name, config
            )
        # Plan 2.8d fix: read batch_size from THIS trial's params, not the
        # fixed base config -- otherwise searching over batch_size has no effect.
        batch_size = current_params["batch_size"]
        q_hard_threshold = float(current_params.get("q_hard_threshold", DEFAULT_Q_HARD_THRESHOLD))
        max_fold_train_seconds = float(
            current_params.get("max_fold_train_seconds", DEFAULT_MAX_FOLD_TRAIN_SECONDS)
        )

        # Record the fields the one-SE selection rule (plan 5.7) needs to rank
        # candidates by simplicity, regardless of whether they were searched
        # or came from the fixed base config.
        trial.set_user_attr("network_depth", current_params.get("network_depth"))
        trial.set_user_attr("network_width", current_params.get("network_width"))
        trial.set_user_attr("l2_penalty", current_params.get("l2_penalty"))

        fold_scores: List[float] = []
        fold_sizes: List[int] = []
        fold_train_seconds: List[float] = []
        fold_records: List[Dict[str, Any]] = []

        def fail(fold_idx, reason, init_seed=None, shuffle_seed=None, train_result=None):
            trial.set_user_attr("failure_reason", f"fold_{fold_idx}_{reason}")
            fold_records.append(build_fold_record(
                fold_idx, init_seed, shuffle_seed, current_params, float("inf"), train_result,
                failure_reason=reason,
            ))
            trial.set_user_attr("fold_records", fold_records)
            return float("inf")

        # Nested three-way split per outer fold (plan section 5.3):
        # gradient_train_idx/earlystop_val_idx drive this candidate's training
        # and early stopping; score_idx is held out from both and only used
        # for the final unified score below, so the same data never both
        # selects this candidate's epoch and scores the candidate.
        for fold_idx, (gradient_train_idx, earlystop_val_idx, score_idx) in enumerate(
            nested_hpo_fold_split(dataset, n_splits, random_seed), start=1
        ):
            # Plan 5.8: a treatment group with too few samples in any of the
            # three splits is a pre-specified failure condition for this trial.
            insufficient_reason = insufficient_treatment_groups_reason(
                dataset,
                {
                    "gradient_train": gradient_train_idx,
                    "earlystop_val": earlystop_val_idx,
                    "score": score_idx,
                },
            )
            if insufficient_reason is not None:
                return fail(fold_idx, insufficient_reason)

            # Pilot HPO fix request item 2: every trial evaluated on this fold
            # shares the same model-init seed and the same DataLoader shuffle
            # seed, derived only from (random_seed, fold_idx) -- NOT
            # trial.number -- so a trial's score differs because of its
            # hyperparameters, not because of a different random draw.
            fold_init_seed, fold_shuffle_seed = fold_control_seeds(random_seed, fold_idx)
            set_random_seed(fold_init_seed)

            trainer = _create_trainer(
                data_configs, current_params, random_seed, dump_folder=None, data_name=data_name
            )

            shuffle_generator = torch.Generator().manual_seed(fold_shuffle_seed)
            train_loader = DataLoader(
                dataset=Subset(dataset, gradient_train_idx.tolist()),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=shuffle_generator,
            )
            earlystop_loader = DataLoader(
                dataset=Subset(dataset, earlystop_val_idx.tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            try:
                model, train_result = trainer.train(train_loader, earlystop_loader)
                # Plan 5.8: a fold that never got a valid checkpoint is a failure.
                if train_result.stop_reason == "no_valid_checkpoint":
                    return fail(fold_idx, "no_valid_checkpoint", fold_init_seed, fold_shuffle_seed, train_result)
                # Pilot HPO fix request item 3: NaN/Inf seen during training
                # itself (not just the final score) is a failure.
                if train_result.n_nonfinite_train > 0 or train_result.n_nonfinite_val > 0:
                    return fail(fold_idx, "nonfinite_during_training", fold_init_seed, fold_shuffle_seed, train_result)

                q_by_treatment = _q_predictions_by_treatment(trainer, model, dataset, score_idx.tolist())
                max_q = _max_abs_q(q_by_treatment)
                # Pilot HPO fix request item 3: Q exceeding a pre-configured
                # hard threshold is a failure, not a value to average in.
                if max_q > q_hard_threshold:
                    return fail(fold_idx, f"q_exceeds_hard_threshold(max_q={max_q:.2f})",
                                fold_init_seed, fold_shuffle_seed, train_result)
                # Pilot HPO fix request item 3: a pre-specified per-fold time
                # budget this fold exceeded (checked after the fact -- this
                # training loop is synchronous, so this cannot pre-empt a
                # run already in progress, only flag it as a failure once done).
                if train_result.train_seconds > max_fold_train_seconds:
                    return fail(fold_idx, f"timeout(train_seconds={train_result.train_seconds:.1f})",
                                fold_init_seed, fold_shuffle_seed, train_result)

                # Unified V-statistic score (plan 5.4), on the untouched score fold.
                fold_score, fold_n = _compute_unified_q_score(
                    trainer, model, dataset, score_idx.tolist()
                )
                # Plan 5.8: NaN/Inf score is a failure, not a value to average in.
                if not math.isfinite(fold_score):
                    return fail(fold_idx, "nonfinite_score", fold_init_seed, fold_shuffle_seed, train_result)

                fold_scores.append(fold_score)
                fold_sizes.append(fold_n)
                fold_train_seconds.append(train_result.train_seconds)
                fold_records.append(build_fold_record(
                    fold_idx, fold_init_seed, fold_shuffle_seed, current_params, fold_score, train_result,
                    failure_reason=None, extra=_q_distribution_stats(q_by_treatment),
                ))

            except Exception as e:
                print(f"[Optuna] Trial {trial.number} Fold {fold_idx} 失败: {e}")
                return fail(fold_idx, f"exception: {e}", fold_init_seed, fold_shuffle_seed)

            finally:
                if 'model' in locals():
                    del model
                if 'trainer' in locals():
                    del trainer
                del train_loader, earlystop_loader
                empty_device_cache(get_device())

        summary_score = weighted_fold_mean(fold_scores, fold_sizes)
        fold_sd, fold_se = fold_score_dispersion(fold_scores)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_sizes", fold_sizes)
        trial.set_user_attr("fold_sd", fold_sd)
        trial.set_user_attr("fold_se", fold_se)
        trial.set_user_attr("total_train_seconds", sum(fold_train_seconds))
        trial.set_user_attr("fold_records", fold_records)
        return summary_score


    print(
        f"[Optuna] 开始 {n_trials} 次试验的 K-fold (k={n_splits}) 调优..."
    )
    effective_tuning_seed = tuning_seed if tuning_seed is not None else random_seed
    if existing_trial_count > 0:
        # Resumed run: derive a NEW, reproducible sampler seed instead of
        # replaying the exact same TPESampler from scratch (see docstring).
        effective_tuning_seed = make_seed_bundle(effective_tuning_seed, existing_trial_count).tuning_seed
    sampler = optuna.samplers.TPESampler(seed=effective_tuning_seed)
    if storage is not None:
        study = optuna.create_study(
            direction="minimize", sampler=sampler, storage=storage,
            study_name=study_name or "nmmr_q_round1", load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"\n[Optuna] 调优完成。")
    # 精简与config管理.md Phase A item 1/2: 在选择最佳 trial 之前先检查是否
    # 存在任何有限值 trial -- 否则下面的 one-SE 选择会在一个 value=inf 的
    # trial 上"成功"跑完,打印出看起来正常但毫无意义的"最佳参数"。
    if not finite_valued_trials(study):
        reason_counts = summarize_trial_failure_reasons(study.trials)
        print(f"[Optuna] Round 1: 全部 {len(study.trials)} 个 trial 失败"
              f"(非有限值/未完成),不选择任何参数,不进入 Round 2。失败原因统计:")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")
        raise NoFiniteRound1TrialError(
            f"Round 1 produced no finite-valued trial out of {len(study.trials)} total."
        )

    # Plan 5.7: the one-SE rule, not the pure minimum, is the primary selection
    # rule; the pure-minimum trial is kept only as audit info alongside it.
    selected_trial, best_trial = select_one_se_candidate(study)
    print(f"  纯最小值 Trial(仅作审计): {best_trial.number}, "
          f"score={best_trial.value:.6f}, params={best_trial.params}")
    print(f"  One-SE 规则选中的 Trial: {selected_trial.number}")
    print(f"  最佳 q-loss (unified V-statistic score): {selected_trial.value:.6f}")
    print(f"  fold_sd={selected_trial.user_attrs.get('fold_sd')}, "
          f"fold_se={selected_trial.user_attrs.get('fold_se')}")
    print(f"  最佳参数: {selected_trial.params}")

    # 合并并返回最终的最优参数(按 one-SE 规则选中的候选,而非纯最小值候选)
    final_best_params = {**base_train_params, **selected_trial.params}

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("\n[Refit] Training the final model on the full dataset using the best params...")

        batch_size = final_best_params['batch_size']
        # Carve out a genuine held-out split instead of reusing the full dataset as
        # both train and val (that would let the "validation" set leak straight
        # into the gradient-trained data and make early stopping/scheduling
        # meaningless).
        all_indices = list(range(len(dataset)))
        full_train_loader, full_val_loader = _build_loaders_for_fold(
            dataset=dataset,
            indices=all_indices,
            batch_size=batch_size,
            seed=random_seed,
        )

        refit_dump_folder = log_folder if log_folder else None

        final_trainer = _create_trainer(
            data_configs=data_configs,
            train_params=final_best_params, # use the selected best hyperparameters
            random_seed=random_seed,
            dump_folder=refit_dump_folder,
            data_name=data_name
        )

        final_model, _final_train_result = final_trainer.train(full_train_loader, full_val_loader, verbose=1)

        stat_type = "u" if str(final_best_params.get('use_u_statistic', 'false')).lower() == 'true' else "v"
        model_filename = f"nmmr_q_{data_name}_{stat_type}_s{scenario}.pth"     
        torch.save(final_model.state_dict(), save_path / model_filename)
                   
        print(f"[Save] 模型已保存: {save_path / model_filename}")

    if return_study:
        return final_best_params, study, selected_trial, best_trial
    return final_best_params

def _run_cross_fitting(data_name:str,
                       dataset,
                       n_splits: int,
                       train_params: Dict[str, Any],
                       data_configs: Dict[str, Any],
                       random_seed: int,
                       log_folder: Optional[Path],
                       tag: str) -> Tuple[torch.Tensor, List[float], np.ndarray, List[Dict[str, Any]]]:
    """
    对任意数据集执行 cross fitting:
    - 将数据划分为 n_splits 折
    - 在每折上训练模型（fold 内部再 8:2 划分 train/val）
    - 在其余折上直接计算 ATE_i，最后求均值

    Returns a 4th value `fold_diagnostics`: one dict per fold with
    `best_epoch`/`stopped_early`/`stop_reason`/`train_seconds` (from that
    fold's TrainResult) plus `moment_risk`/`n_eval` -- the unified
    V-statistic held-out moment risk (same scorer HPO uses, see plan 5.4),
    evaluated on this fold's held-out eval split using the kernel bandwidth
    already frozen from this fold's own gradient-training subset. This is
    what plan 6.1/hpo_selection_plan.md section 9's pilot-confirm diagnostics
    ("矩风险"、"所有折都恢复了最佳模型"、"训练时间") read from.
    """
    batch_size = train_params['batch_size']
    cf_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + 1024)

    # 存储全量预测结果 (N, 1)
    dataset_len = len(dataset)
    q_preds_full = np.zeros((dataset_len, 1), dtype=np.float32)

    fold_predictions: List[torch.Tensor] = []
    ate_values: List[float] = []
    fold_diagnostics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, eval_idx) in enumerate(cf_kfold.split(np.arange(len(dataset))), start=1):
        print(f"\n[CF-{tag}] Fold {fold_idx}/{n_splits}")
        train_fold_indices = train_idx.tolist()   # 当前折用于训练
        eval_fold_indices = eval_idx.tolist()   # 剩余样本用于计算 ATE

        fold_folder = None
        if log_folder is not None:
            fold_folder = log_folder / f"fold_{fold_idx}"
            fold_folder.mkdir(parents=True, exist_ok=True)

        # Explicitly derive this fold's init seed from (random_seed, fold_idx) so
        # model initialization is tied to the fold rather than to however much of
        # the global RNG stream happened to be consumed by earlier folds/trials.
        fold_init_seed = make_seed_bundle(random_seed, fold_idx).init_seed
        set_random_seed(fold_init_seed)

        trainer = _create_trainer(data_configs, train_params, random_seed, fold_folder, data_name)

        train_loader, val_loader = _build_loaders_for_fold(
            dataset=dataset,
            indices=train_fold_indices,
            batch_size=batch_size,
            seed=random_seed + fold_idx,
        )
        model, fold_train_result = trainer.train(train_loader, val_loader)
        print(f"  Fold {fold_idx} train result: stop_reason={fold_train_result.stop_reason}, "
              f"best_epoch={fold_train_result.best_epoch}, best_val_score={fold_train_result.best_val_score}")

        eval_view = _build_dataset_view(dataset, eval_fold_indices) 
        fold_ate = trainer.predict(model, eval_view)
        fold_predictions.append(fold_ate)
        ate_values.append(fold_ate.item())
        print(f"  Fold {fold_idx} ATE: {fold_ate.item():.4f}")
        
        with torch.no_grad():
            Z = eval_view.Z.to(trainer.device)
            X = eval_view.X.to(trainer.device)
            A = eval_view.A.to(trainer.device)
            inputs = torch.cat((Z, X), dim=1)
            
            # 分别预测 q(Z,0,X) 和 q(Z,1,X)
            # 注意: 这里的 model 是 NMMR_Q_DualModel
            q0 = model.net0(inputs)
            q1 = model.net1(inputs)
            
            # 根据实际 A 选择 q 值
            q_pred = torch.where(A > 0.5, q1, q0)
            
            # 存入全量数组 (注意索引对应)
            q_preds_full[eval_fold_indices] = q_pred.cpu().numpy()

        moment_risk, n_eval = _compute_unified_q_score(trainer, model, dataset, eval_fold_indices)
        fold_diagnostics.append({
            "fold": fold_idx,
            "moment_risk": moment_risk,
            "n_eval": n_eval,
            "best_epoch": fold_train_result.best_epoch,
            "stopped_early": fold_train_result.stopped_early,
            "stop_reason": fold_train_result.stop_reason,
            "train_seconds": fold_train_result.train_seconds,
        })

        empty_device_cache(trainer.device)
        del model

    if not fold_predictions:
        raise RuntimeError("Cross-fitting 阶段未生成任何预测结果。")

    ate_tensor = torch.stack(fold_predictions).mean()
    print(f"[CF-{tag}] 平均 ATE(PIPW): {ate_tensor.item():.6f}")
    return ate_tensor, ate_values, q_preds_full, fold_diagnostics

def _run_traditional_prediction(
    dataset,
    train_params: Dict[str, Any],
    model_path: Path,
    device: str
) -> Tuple[float, np.ndarray]:
    """
    Loads a trained model and predicts on the given dataset without cross-fitting.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    print(f"[Traditional] Loading model from {model_path}")
    
    Z_dim = dataset.Z.shape[1]
    X_dim = dataset.X.shape[1]
    input_size = Z_dim + X_dim
    
    model0 = NMMR_Q_common(input_dim=input_size, train_params=train_params)
    model1 = NMMR_Q_common(input_dim=input_size, train_params=train_params)
    
    dual_model = NMMR_Q_DualModel(model0, model1)
    
    # 2. Load Weights
    state_dict = torch.load(model_path, map_location=device)
    dual_model.load_state_dict(state_dict)
    dual_model.to(device)
    dual_model.eval()

    full_view = _build_dataset_view(dataset, list(range(len(dataset))))

    ate_val = NMMR_Q_Trainer.predict(dual_model, full_view)

    with torch.no_grad():
        Z = full_view.Z.to(device)
        X = full_view.X.to(device)
        A = full_view.A.to(device)
        
        q_hat = dual_model(Z, X, A)
        q_preds_full = q_hat.cpu().numpy()

    return ate_val.item(), q_preds_full

def NMMR_Q_experiment(dataset_name: str,
                      data_configs: Dict[str, Any],
                      train_params: Dict[str, Any],
                      log_folder: Path,
                      random_seed: int,
                      dataset=None,
                      mode: str = 'cross_fitting',
                      model_dir: Optional[Path] = None,
                      data_seed: Optional[int] = None,
                      init_seed: Optional[int] = None,
                      tuning_seed: Optional[int] = None,
                      ):
    """
    运行 NMMR-Q 实验 (求解 q 函数并计算 ATE)
    Args:
        mode: 'cross_fitting' or 'traditional'
        model_path: Path to the .pth file (required if mode='traditional')
        random_seed: 本次重复的 split_seed，控制交叉拟合分折
        data_seed: 本次重复的 data_seed，控制 SGD 数据生成；为 None 时回退到 random_seed
        init_seed: 本次重复的 init_seed，控制模型初始化；为 None 时回退到 random_seed
        tuning_seed: 本次重复的 tuning_seed，控制 Optuna 采样；为 None 时回退到 random_seed
    """
    data_seed = data_seed if data_seed is not None else random_seed
    init_seed = init_seed if init_seed is not None else random_seed
    tuning_seed = tuning_seed if tuning_seed is not None else random_seed

    device = get_device()
    print(f"使用设备: {device}")

    dataset_key = dataset_name.lower()
      
    if mode == 'traditional':
        if model_dir is None:
            raise ValueError("In 'traditional' mode, 'model_path' must be provided.")
            
        ate_val, q_preds = _run_traditional_prediction(
            dataset=dataset,
            train_params=train_params, 
            model_path=model_dir,
            device=device
        )
        return {
            "PIPW": ate_val,
            "q_preds": q_preds
        }
    
    train_params_copy = copy.deepcopy(train_params)
    base_params, search_space = _parse_optuna_space(train_params_copy)
    n_splits = int(base_params.get('n_splits', 5))

    # NMMR_Q_experiment is the formal/fixed-parameter entrypoint: it no
    # longer launches Optuna inline (pilot HPO fix request item 8).
    # Hyperparameter search happens exclusively in
    # scripts/run_pilot_protocol.py (which calls _run_optuna_tuning
    # directly, not through this function) and ends in a frozen config.
    if search_space:
        raise ValueError(
            f"NMMR_Q_experiment no longer runs HPO inline; train_params still "
            f"has an Optuna search space for {list(search_space)}. "
            "Run scripts/run_pilot_protocol.py first and provide a frozen config."
        )

    if dataset is not None:
        set_random_seed(init_seed)
        ate_estimate, ate_values, q_preds, fold_diagnostics = _run_cross_fitting(
            data_name=dataset_key,
            dataset=dataset,
            n_splits=n_splits,
            train_params=base_params,
            data_configs=data_configs,
            random_seed=random_seed,
            log_folder=log_folder,
            tag="cross_fitting",
        )
        # 返回结果供 double_rob.py 使用
        return {
            "PIPW": ate_estimate.item(),
            "q_preds": q_preds,
            "fold_diagnostics": fold_diagnostics,
        }
    
    base_params.pop('hpo_n_trials', None)  # HPO-only metadata, unused now that this function no longer tunes.
    final_train_params: Dict[str, Any] = {}


    if dataset_key == 'sgd':
        
        data_base_path = Path(data_configs['data_path'])
        n_trials = data_configs.get('n_trials',-1)
        n_samples = data_configs.get('n_samples',2000)
        scenario = data_configs.get('scenario', 1)
        # 重复实验的时候不写入数据
        if n_trials > 0:
            PIPW = []
            use_u = base_params['use_u_statistic']
            loss_name = "u" if use_u == "true" else "v"
            for trial_idx in range(1, n_trials + 1):
                trial_seeds = make_seed_bundle(data_seed, trial_idx)
                df = generate_data(n_samples=n_samples, scenario=scenario, seed=trial_seeds.data_seed)
                dataset_trial = SGDDataset(csv_path='', df=df, device=device)
                set_random_seed(trial_seeds.init_seed)
                ate_estimate, ate_values, _, _ = _run_cross_fitting(
                    data_name=dataset_key,
                    dataset=dataset_trial,
                    n_splits=n_splits,
                    train_params=base_params,
                    data_configs=data_configs,
                    random_seed=trial_seeds.split_seed,
                    log_folder=log_folder,
                    tag="sgd_cf",
                )
                PIPW.append(ate_estimate.item())
            results = pd.DataFrame({
                "PIPW": PIPW
            })
            print(f"\n--- SGD ATE 估计结果 (平均 over {n_trials} 次试验):")
            print(f"PIPW = {results['PIPW'].mean():.4f}")
            print("----------------------------------")
            
            output_path = data_configs.get('output_path', 'predicts/sgd/nmmr')
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / f"pipw_{loss_name}_s{scenario}.csv"
            results.to_csv(out_file, index=False)

        else:
            train_data_seeds = make_seed_bundle(data_seed, 0)
            cf_data_seeds = make_seed_bundle(data_seed, 1)
            train_df = generate_data(n_samples=n_samples, scenario=scenario, seed=train_data_seeds.data_seed)
            cf_df = generate_data(n_samples=n_samples, scenario=scenario, seed=cf_data_seeds.data_seed)

            output_dir = data_base_path
            output_dir.mkdir(parents=True, exist_ok=True)
            sgd_train = output_dir / 'sgd_train.csv'
            sgd_cf = output_dir / 'sgd_cf.csv'

            train_df.to_csv(sgd_train, index=False)
            cf_df.to_csv(sgd_cf, index=False)
        
            train_path = _locate_split_file(
                data_base_path,
                data_configs.get('train_file'),
                ['sgd_train.csv', 'train.csv'],
            )
            if train_path is None:
                raise FileNotFoundError(f"未在 {data_base_path} 找到 SGD 训练数据。")

            cf_path = _locate_split_file(
                data_base_path,
                data_configs.get('cf_file'),
                ['sgd_cf.csv', 'sgd_test.csv', 'test.csv'],
            )
            if cf_path is None:
                raise FileNotFoundError(f"未在 {data_base_path} 找到 cross-fitting 数据 (例如 sgd_cf.csv)。")

            cf_dataset = SGDDataset(cf_path, device=device)

            print("\n===> 使用 config 中的固定参数")
            final_train_params = base_params

            print("\n===> 在 SGD CF 数据集上执行 cross fitting (使用最优/固定参数)")
            set_random_seed(init_seed)
            ate_estimate, ate_values, _, _ = _run_cross_fitting(
                data_name=dataset_key,
                dataset=cf_dataset,
                n_splits=n_splits,
                train_params=final_train_params,
                data_configs=data_configs,
                random_seed=random_seed,
                log_folder=log_folder,
                tag="sgd_cf",
            )

    elif dataset_key == 'rhc':
        rhc_data_dir = data_configs.get('data_path')
        use_all_x = data_configs.get('use_all_X', False)
        trainer_data_configs = {'rhc': data_configs}
        
        print(f"加载 RHC 数据集 (use_all_X={use_all_x})...")
        
        train_dataset = RHCDataset(split='train', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
        print("\n===> [RHC] 使用 config 中的固定参数")
        final_train_params = base_params

        print(f"\n===> [RHC] 加载 Val/Test 集并合并，准备执行 Cross Fitting")
        val_dataset = RHCDataset(split='val', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
        test_dataset = RHCDataset(split='test', use_all_x=use_all_x, data_dir=rhc_data_dir, device=device)
        full_dataset = MergedDataset([train_dataset, val_dataset, test_dataset])

        print(f"     全量数据样本数: {len(full_dataset)}")

        set_random_seed(init_seed)
        ate_estimate, ate_values, _, _ = _run_cross_fitting(
            data_name=dataset_key,
            dataset=full_dataset, # 在全量数据上进行 Cross Fitting
            n_splits=n_splits,
            train_params=final_train_params,
            data_configs=trainer_data_configs,
            random_seed=random_seed,
            log_folder=log_folder,
            tag="rhc_full_cf",
        )
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    print(f"\n--- 最终 ATE(PIPW) (Seed {random_seed}) ---")
    print(f"ATE(PIPW) = {ate_estimate.item():.4f}")
    print("----------------------------------")


def train_q_standalone(data_configs, train_params, dataset_name, scenario, save_dir, log_folder, random_seed=42,
                        data_seed=None, init_seed=None, tuning_seed=None):

    """
    独立训练模式：
    1. 加载/生成数据
    2. (可选) 使用 Optuna 进行 K-Fold 调参
    3. 使用最佳参数在全量数据上训练
    4. 保存模型权重和最佳参数配置
    """
    data_seed = data_seed if data_seed is not None else random_seed
    init_seed = init_seed if init_seed is not None else random_seed
    tuning_seed = tuning_seed if tuning_seed is not None else random_seed

    device = get_device()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[Standalone] 启动独立训练流程 | 数据集: {dataset_name} | Scenario: {scenario}")

    dataset_key = dataset_name.lower()
    if dataset_key == 'sgd':
        n_samples = data_configs.get('n_samples',2000)
        df = generate_data(n_samples=n_samples, scenario=scenario, seed=data_seed)
        # 这里的 csv_path 为空，因为是实时生成的
        train_dataset = SGDDataset(csv_path='', df=df, device=device)
        
    elif dataset_key == 'rhc':
        print(f"[Standalone_q] 加载 RHC 训练数据...")
        # 注意：RHC通常需要读取 data_path，这里假设 data_configs 里有
        data_dir = data_configs.get('data_path')
        use_all_x = data_configs.get('use_all_X', False)
        train_dataset = RHCDataset(split='train', use_all_x=use_all_x, data_dir=data_dir, device=device)
    else:
        raise ValueError(f"未知的 dataset_name: {dataset_name}")
    
    
    train_params_copy = copy.deepcopy(train_params)
    base_params, search_space = _parse_optuna_space(train_params_copy)
    n_splits = int(base_params.get('n_splits', 5))

    hpo_n_trials = int(base_params.pop('hpo_n_trials', 20))

    set_random_seed(init_seed)
    _run_optuna_tuning(
        data_name=dataset_key,
        dataset=train_dataset,
        n_splits=n_splits,
        n_trials=hpo_n_trials,
        base_train_params=base_params,
        search_space=search_space,
        data_configs=data_configs,
        random_seed=random_seed,
        log_folder=log_folder,
        save_dir=save_dir,
        scenario=scenario,
        tuning_seed=tuning_seed,
    )