import torch
import numpy as np
import pandas as pd
import copy
import math
import optuna
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


from src.models.NMMRh.nmmr_h_trainers import NMMR_H_Trainer, NMMR_H_Trainer_SGD, NMMR_H_Trainer_RHC, MLP_for_NMMR
from src.models.NMMRh.nmmr_h_model import MLP_for_NMMR
from src.models.NMMRh.nmmr_h_loss import NMMR_H_Loss
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

def _create_trainer(data_configs, train_params, random_seed, dump_folder, data_name):
    """
    根据配置创建对应的 H Trainer
    """
    if data_name == 'sgd':
        return NMMR_H_Trainer_SGD(
            data_configs=data_configs,
            train_params=train_params,
            random_seed=random_seed,
            dump_folder=dump_folder,
        )
    elif data_name == 'rhc' :
        return NMMR_H_Trainer_RHC(
            data_configs=data_configs,
            train_params=train_params,
            random_seed=random_seed,
            dump_folder=dump_folder,
        )

def _compute_unified_h_score(trainer, model: torch.nn.Module, dataset, indices: List[int]) -> Tuple[float, int]:
    """
    Unified HPO scoring metric for H (plan section 5.4): regardless of
    whether this candidate trained with the U- or V-statistic objective,
    candidates are always compared on the V-statistic moment risk
        S_H(lambda) = (1/n^2) r^T K_V r,
    evaluated on `indices` -- a held-out "score" fold that must never have
    participated in this candidate's gradient training or early stopping
    (see nested_hpo_fold_split). Uses the fixed reference kernel (median
    heuristic fitted on the trainer's gradient-training subset, multiplier
    fixed at 1), NOT the candidate's own training kernel, so scores are
    comparable across candidates with different kernel_bandwidth_multiplier.
    """
    view = _build_dataset_view(dataset, indices)
    scorer = NMMR_H_Loss(use_u_statistic=False)
    with torch.no_grad():
        A = view.A.to(trainer.device)
        W = view.W.to(trainer.device)
        Z = view.Z.to(trainer.device)
        X = view.X.to(trainer.device)
        Y = view.Y.to(trainer.device)

        k_matrix = trainer._compute_reference_kernel_matrix(A, Z, X)

        model_input = torch.cat((A, W, X), dim=1)
        pred = model(model_input)

        score = scorer(pred, Y, k_matrix)

    return float(score.item()), len(indices)

def _run_optuna_tuning(
    data_name:str,
    dataset,
    n_splits: int,
    n_trials: int,
    base_train_params: Dict[str, Any],
    search_space: Dict[str, Any],
    data_configs: Dict[str, Any],
    random_seed: int,
    log_folder: Optional[Path] = None,
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

    `storage`/`study_name`: when given, the Optuna study is persisted there
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        current_params = copy.deepcopy(base_train_params)
        for param_name, config in search_space.items():
            current_params[param_name] = _suggest_from_trial(
                trial, param_name, config
            )
        # Plan 2.8d fix: read batch_size from THIS trial's params, not the
        # fixed base config -- otherwise searching over batch_size has no effect.
        batch_size = current_params["batch_size"]

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
            t_loader = DataLoader(
                Subset(dataset, gradient_train_idx.tolist()), batch_size=batch_size,
                shuffle=True, num_workers=0, generator=shuffle_generator,
            )
            earlystop_loader = DataLoader(Subset(dataset, earlystop_val_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)

            try:
                model, train_result = trainer.train(t_loader, earlystop_loader)
                # Plan 5.8: a fold that never got a valid checkpoint is a failure.
                if train_result.stop_reason == "no_valid_checkpoint":
                    return fail(fold_idx, "no_valid_checkpoint", fold_init_seed, fold_shuffle_seed, train_result)
                # Pilot HPO fix request item 3: NaN/Inf seen during training
                # itself (not just the final score) is a failure.
                if train_result.n_nonfinite_train > 0 or train_result.n_nonfinite_val > 0:
                    return fail(fold_idx, "nonfinite_during_training", fold_init_seed, fold_shuffle_seed, train_result)

                # Unified V-statistic score (plan 5.4), on the untouched score fold.
                fold_score, fold_n = _compute_unified_h_score(
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
                    failure_reason=None,
                ))
            except Exception as e:
                print(f"[Optuna] Trial {trial.number} Fold {fold_idx} 失败: {e}")
                return fail(fold_idx, f"exception: {e}", fold_init_seed, fold_shuffle_seed)
            finally:
                if 'model' in locals(): del model
                if 'trainer' in locals(): del trainer
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

    print(f"[Optuna] 开始 {n_trials} 次试验的 K-fold (k={n_splits}) 调优...")
    effective_tuning_seed = tuning_seed if tuning_seed is not None else random_seed
    if existing_trial_count > 0:
        # Resumed run: derive a NEW, reproducible sampler seed instead of
        # replaying the exact same TPESampler from scratch (see docstring).
        effective_tuning_seed = make_seed_bundle(effective_tuning_seed, existing_trial_count).tuning_seed
    sampler = optuna.samplers.TPESampler(seed=effective_tuning_seed)
    if storage is not None:
        study = optuna.create_study(
            direction="minimize", sampler=sampler, storage=storage,
            study_name=study_name or "nmmr_h_round1", load_if_exists=True,
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

    selected_trial, best_trial = select_one_se_candidate(study)
    print(f"  纯最小值 Trial(仅作审计): {best_trial.number}, "
          f"score={best_trial.value:.6f}, params={best_trial.params}")
    print(f"  One-SE 规则选中的 Trial: {selected_trial.number}")
    print(f"  最佳 h-loss (unified V-statistic score): {selected_trial.value:.6f}")
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
        model_filename = f"nmmr_h_{data_name}_{stat_type}_s{scenario}.pth"     
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
                       tag: str,
                       fold_plan: Optional[List[Tuple[Any, Any]]] = None,
                       init_seed: Optional[int] = None,
                       test_dataset: Optional[Any] = None,
                       )  -> Tuple[torch.Tensor, List[float], Dict, List[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    """
    Returns a 4th value `fold_diagnostics`: one dict per fold with
    `best_epoch`/`stopped_early`/`stop_reason`/`train_seconds` (from that
    fold's TrainResult) plus `moment_risk`/`n_eval` -- the unified
    V-statistic held-out moment risk (same scorer HPO uses, see plan 5.4),
    evaluated on this fold's held-out eval split using the kernel bandwidth
    already frozen from this fold's own gradient-training subset. This is
    what plan 6.1/hpo_selection_plan.md section 9's pilot-confirm diagnostics
    ("矩风险"、"所有折都恢复了最佳模型"、"训练时间") read from.

    Three optional, purely additive parameters (plan/p0_e1_e2/01_shared_protocol.md
    section 5), all backward-compatible no-ops when omitted -- see the
    identical ones on NMMRq's `_run_cross_fitting` for the full rationale:

    - `fold_plan`: an explicit list of (train_idx, eval_idx) pairs, used
      AS-IS when given (never re-split internally).
    - `init_seed`: the base seed each fold's model initialization derives
      from; omitted means every existing caller's behavior is unchanged
      (still derived from `random_seed`).
    - `test_dataset`: predicted by each fold's model before it's released;
      the 5th return value is `None` when omitted, or
      `{"h0_test": ..., "h1_test": ...}` (equal-weight fold average) when given.
    """
    batch_size = train_params['batch_size']
    if fold_plan is not None:
        fold_iterable = list(enumerate(fold_plan, start=1))
    else:
        cf_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + 1024)
        fold_iterable = list(enumerate(cf_kfold.split(np.arange(len(dataset))), start=1))

    base_init_seed = init_seed if init_seed is not None else random_seed

    fold_predictions: List[torch.Tensor] = []
    ate_values: List[float] = []
    fold_diagnostics: List[Dict[str, Any]] = []

    dataset_len = len(dataset)

    h_fact_full = np.zeros((dataset_len, 1), dtype=np.float32)
    h_1_full = np.zeros((dataset_len, 1), dtype=np.float32)
    h_0_full = np.zeros((dataset_len, 1), dtype=np.float32)
    test_h1_per_fold: List[np.ndarray] = []
    test_h0_per_fold: List[np.ndarray] = []

    for fold_idx, (train_idx_np, eval_idx_np) in fold_iterable:
        print(f"\n[CF-{tag}] Fold {fold_idx}/{n_splits}")
        train_fold_indices = list(train_idx_np)
        eval_fold_indices = list(eval_idx_np)

        fold_folder = None
        if log_folder is not None:
            fold_folder = log_folder / f"fold_{fold_idx}"
            fold_folder.mkdir(parents=True, exist_ok=True)

        # Explicitly derive this fold's init seed from (base_init_seed, fold_idx) so
        # model initialization is tied to the fold rather than to however much of
        # the global RNG stream happened to be consumed by earlier folds/trials.
        fold_init_seed = make_seed_bundle(base_init_seed, fold_idx).init_seed
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

        # 预测 ATE
        eval_view = _build_dataset_view(dataset, eval_fold_indices)
        fold_ate = trainer.predict(model, eval_view)
        fold_predictions.append(fold_ate)
        ate_values.append(fold_ate.item())
        print(f"  Fold {fold_idx} ATE: {fold_ate.item():.4f}")

        with torch.no_grad():
            W = eval_view.W.to(trainer.device)
            X = eval_view.X.to(trainer.device)
            A = eval_view.A.to(trainer.device)

            # Factual h(A, W, X)
            inputs_fact = torch.cat((A, W, X), dim=1)
            h_fact = model(inputs_fact)

            # Counterfactual h(1, W, X)
            inputs_1 = torch.cat((torch.ones_like(A), W, X), dim=1)
            h_1 = model(inputs_1)

            # Counterfactual h(0, W, X)
            inputs_0 = torch.cat((torch.zeros_like(A), W, X), dim=1)
            h_0 = model(inputs_0)

            h_fact_full[eval_fold_indices] = h_fact.cpu().numpy()
            h_1_full[eval_fold_indices] = h_1.cpu().numpy()
            h_0_full[eval_fold_indices] = h_0.cpu().numpy()

        if test_dataset is not None:
            test_view = _build_dataset_view(test_dataset, list(range(len(test_dataset))))
            with torch.no_grad():
                Wt = test_view.W.to(trainer.device)
                Xt = test_view.X.to(trainer.device)
                ones_t = torch.ones((Wt.shape[0], 1), device=trainer.device)
                zeros_t = torch.zeros((Wt.shape[0], 1), device=trainer.device)
                h1_t = model(torch.cat((ones_t, Wt, Xt), dim=1))
                h0_t = model(torch.cat((zeros_t, Wt, Xt), dim=1))
            test_h1_per_fold.append(h1_t.cpu().numpy())
            test_h0_per_fold.append(h0_t.cpu().numpy())

        moment_risk, n_eval = _compute_unified_h_score(trainer, model, dataset, eval_fold_indices)
        fold_diagnostics.append({
            "fold": fold_idx,
            "moment_risk": moment_risk,
            "n_eval": n_eval,
            "best_epoch": fold_train_result.best_epoch,
            "stopped_early": fold_train_result.stopped_early,
            "stop_reason": fold_train_result.stop_reason,
            # Direct signal pilot-confirm's all_folds_restored_best_checkpoint
            # check relies on (精简与config管理.md Phase C follow-up) -- not
            # inferred from stop_reason, which stays only as a diagnostic.
            "best_state_restored": fold_train_result.best_state_restored,
            "train_seconds": fold_train_result.train_seconds,
        })

        empty_device_cache(trainer.device)
        del model

    if not fold_predictions:
        raise RuntimeError("Cross-fitting 阶段未生成任何预测结果。")

    ate_tensor = torch.stack(fold_predictions).mean()
    print(f"[CF-{tag}] 平均 ATE(POR): {ate_tensor.item():.6f}")

    test_predictions = None
    if test_dataset is not None:
        # Equal-weight average across folds (plan/p0_e1_e2/01_shared_protocol.md
        # section 5: "由于每个模型训练样本数相同，等权规则预先固定").
        test_predictions = {
            "h1_test": np.mean(np.stack(test_h1_per_fold, axis=0), axis=0).reshape(-1),
            "h0_test": np.mean(np.stack(test_h0_per_fold, axis=0), axis=0).reshape(-1),
        }

    return ate_tensor, ate_values, {
        "h_fact": h_fact_full,
        "h_1": h_1_full,
        "h_0": h_0_full
    }, fold_diagnostics, test_predictions

def _run_traditional_prediction(
    dataset,
    train_params: Dict[str, Any],
    model_path: Path,
    device: str
) -> Tuple[torch.Tensor, Dict]:
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    
    print(f"[Traditional] Loading model from {model_path}")
    

    W_dim = dataset.W.shape[1]
    X_dim = dataset.X.shape[1]
    input_size = 1 + W_dim + X_dim  
    
    model = MLP_for_NMMR(input_dim = input_size, train_params=train_params)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    full_view = _build_dataset_view(dataset, list(range(len(dataset))))
    
    ate_val = NMMR_H_Trainer.predict(model, full_view)
    
    with torch.no_grad():
            W = full_view.W.to(device)
            X = full_view.X.to(device)
            A = full_view.A.to(device)
            
            # Factual h(A, W, X)
            inputs_fact = torch.cat((A, W, X), dim=1)
            h_fact = model(inputs_fact)
            
            # Counterfactual h(1, W, X)
            inputs_1 = torch.cat((torch.ones_like(A), W, X), dim=1)
            h_1 = model(inputs_1)
            
            # Counterfactual h(0, W, X)
            inputs_0 = torch.cat((torch.zeros_like(A), W, X), dim=1)
            h_0 = model(inputs_0)
            
            h_fact_full = h_fact.cpu().numpy()
            h_1_full = h_1.cpu().numpy()
            h_0_full = h_0.cpu().numpy()
            
    return ate_val, {
        "h_fact": h_fact_full,
        "h_1": h_1_full,
        "h_0": h_0_full
    }
# -------------------------------------------------------
# 主入口函数: NMMR_H_experiment
# -------------------------------------------------------

def NMMR_H_experiment(dataset_name: str,
                      data_configs: Dict[str, Any],
                      train_params: Dict[str, Any],
                      log_folder: Path,
                      random_seed: int,
                      dataset = None,
                      mode: str = 'cross_fitting',
                      model_dir: Optional[Path] = None,
                      data_seed: Optional[int] = None,
                      init_seed: Optional[int] = None,
                      tuning_seed: Optional[int] = None,
                      ):
    """
    运行 NMMR-H 实验 (求解 h 函数并计算 ATE)
    Args:
        random_seed: 本次重复的 split_seed，控制交叉拟合分折
        data_seed: 本次重复的 data_seed，控制 SGD 数据生成；为 None 时回退到 random_seed
        init_seed: 本次重复的 init_seed，控制模型初始化；为 None 时回退到 random_seed
        tuning_seed: 本次重复的 tuning_seed，控制 Optuna 采样；为 None 时回退到 random_seed
    """
    data_seed = data_seed if data_seed is not None else random_seed
    init_seed = init_seed if init_seed is not None else random_seed
    tuning_seed = tuning_seed if tuning_seed is not None else random_seed

    device = get_device()

    dataset_key = dataset_name.lower()
    
    if mode == 'traditional':
        if model_dir is None:
            raise ValueError("In 'traditional' mode, 'model_path' must be provided.")
            
        ate_val, h_preds = _run_traditional_prediction(
            dataset=dataset,
            train_params=train_params, 
            model_path=model_dir,
            device=device
        )
        return {
            "POR": ate_val.item(),
            "h_preds": h_preds
        }
    
    train_params_copy = copy.deepcopy(train_params)
    base_params, search_space = _parse_optuna_space(train_params_copy)
    n_splits = int(base_params.get('n_splits', 5))

    # NMMR_H_experiment is the formal/fixed-parameter entrypoint: it no
    # longer launches Optuna inline (pilot HPO fix request item 8).
    # Hyperparameter search happens exclusively in
    # scripts/run_pilot_protocol.py (which calls _run_optuna_tuning
    # directly, not through this function) and ends in a frozen config.
    if search_space:
        raise ValueError(
            f"NMMR_H_experiment no longer runs HPO inline; train_params still "
            f"has an Optuna search space for {list(search_space)}. "
            "Run scripts/run_pilot_protocol.py first and provide a frozen config."
        )

    if dataset is not None:
        set_random_seed(init_seed)
        ate_estimate, ate_values, h_preds, fold_diagnostics, _ = _run_cross_fitting(
            data_name=dataset_key,
            dataset=dataset,
            n_splits=n_splits,
            train_params=base_params,
            data_configs=data_configs,
            random_seed=random_seed,
            log_folder=log_folder,
            tag="external_cf",
        )
        return {
            "POR": ate_estimate.item(),
            "h_preds": h_preds,
            "fold_diagnostics": fold_diagnostics,
        }
    
    base_params.pop('hpo_n_trials', None)  # HPO-only metadata, unused now that this function no longer tunes.
    final_train_params: Dict[str, Any] = {}

    
    if dataset_key == 'sgd':
        data_base_path = Path(data_configs['data_path'])
        n_trials = data_configs.get('n_trials',-1)
        n_samples = data_configs.get('n_samples',2000)
        scenario = data_configs.get('scenario', 1) 

        if n_trials > 0:
            POR = []
            use_u = base_params['use_u_statistic']
            loss_name = "u" if use_u == "true" else "v"
            for trial_idx in range(1, n_trials + 1):
                trial_seeds = make_seed_bundle(data_seed, trial_idx)
                df = generate_data(n_samples=n_samples, scenario=scenario, seed=trial_seeds.data_seed)
                dataset_trial = SGDDataset(csv_path='', df=df, device=device)
                set_random_seed(trial_seeds.init_seed)
                ate_estimate, ate_values, _, _, _ = _run_cross_fitting(
                    data_name=dataset_key,
                    dataset=dataset_trial,
                    n_splits=n_splits,
                    train_params=base_params,
                    data_configs=data_configs,
                    random_seed=trial_seeds.split_seed,
                    log_folder=log_folder,
                    tag="sgd_cf",
                )
                POR.append(ate_estimate.item())
            results = pd.DataFrame({
                "POR": POR
            })
            print(f"\n--- SGD ATE 估计结果 (平均 over {n_trials} 次试验):")
            print(f"POR = {results['POR'].mean():.4f}")
            print("----------------------------------")
            
            output_path = data_configs.get('output_path', 'predicts/sgd/nmmr_q')
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / f"por_{loss_name}_s{scenario}.csv"
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
            print("\n===> 使用固定默认参数")
            final_train_params = base_params

            print("\n===> 在 SGD CF 数据集上执行 cross fitting (使用最优/固定参数)")
            set_random_seed(init_seed)
            ate_estimate, ate_values, _, _, _ = _run_cross_fitting(
                data_name=dataset_key,
                dataset=cf_dataset,
                n_splits=n_splits,
                train_params=final_train_params,
                data_configs=data_configs,
                random_seed=random_seed,
                log_folder=log_folder,
                tag="sgd_test",
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
        ate_estimate, ate_values, _, _, _ = _run_cross_fitting(
            data_name=dataset_key,
            dataset=full_dataset,
            n_splits=n_splits,
            train_params=final_train_params,
            data_configs=trainer_data_configs,
            random_seed=random_seed,
            log_folder=log_folder,
            tag="rhc_full",
        )

    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    print(f"\n--- 最终 ATE(POR) (Seed {random_seed}) ---")
    print(f"ATE(POR) = {ate_estimate.item():.4f}")
    print("----------------------------------")

def train_h_standalone(data_configs, train_params, dataset_name, scenario, save_dir, log_folder, random_seed=42,
                        data_seed=None, init_seed=None, save_model=False):
    """
    Standalone (non-cross-fitting) training path, kept as a horizontal
    comparison point against cross-fitting (精简与config管理.md section
    6.6 / Phase D). Only reads an already-frozen `train_params` (no search
    space -- Optuna runs exclusively in scripts/run_pilot_protocol.py now,
    this function never launches it), trains ONE model on a single
    train/test split with those same frozen parameters, and reports a
    non-cross-fitting effect estimate, held-out test error (the same
    unified moment-risk scorer used everywhere else in this pipeline), and
    the total wall-clock runtime. Model weights are only written to
    `save_dir` when `save_model=True` -- never by default, so Monte Carlo
    repetitions don't accumulate one checkpoint per repetition.
    """
    data_seed = data_seed if data_seed is not None else random_seed
    init_seed = init_seed if init_seed is not None else random_seed

    device = get_device()

    print(f"\n[Standalone_h] 启动独立(非 cross-fitting)训练流程 | 数据集: {dataset_name} | Scenario: {scenario}")

    dataset_key = dataset_name.lower()
    if dataset_key == 'sgd':
        n_samples = data_configs.get('n_samples', 2000)
        df = generate_data(n_samples=n_samples, scenario=scenario, seed=data_seed)
        dataset = SGDDataset(csv_path='', df=df, device=device)

    elif dataset_key == 'rhc':
        # Phase D explicit scope limit (精简与config管理.md section 6.6), not an
        # oversight: _run_cross_fitting's RHC population is the merged
        # train+val+test set, while a single RHCDataset(split='train') here
        # would be a smaller, different population -- not a valid horizontal
        # comparison against cross-fitting. Revisit once RHC's standalone
        # split is unified with cross-fitting's population.
        raise NotImplementedError(
            "train_h_standalone 目前只支持 dataset_name='sgd'。RHC 的 standalone 数据 "
            "划分与 cross-fitting 使用的全量 RHC 数据(train+val+test 合并)不一致，"
            "暂不提供可横向比较的 RHC standalone 流程。"
        )
    else:
        raise ValueError(f"未知的 dataset_name: {dataset_name}")

    # train_h_standalone only ever consumes an already-frozen config (same
    # invariant NMMR_H_experiment enforces for its own entrypoint): a
    # leftover Optuna search space here is a config error, not something to
    # tune around.
    _, search_space = _parse_optuna_space(copy.deepcopy(train_params))
    if search_space:
        raise ValueError(
            f"train_h_standalone no longer runs HPO inline; train_params still has "
            f"an Optuna search space for {list(search_space)}. "
            "Run scripts/run_pilot_protocol.py first and provide a frozen config."
        )

    n_splits = int(train_params.get('n_splits', 5))
    batch_size = train_params['batch_size']

    start_time = time.time()
    set_random_seed(init_seed)

    # Reuses the same outer KFold convention _run_cross_fitting uses (same
    # seed offset, same 1/n_splits held-out fraction) but takes only its
    # first split -- a single train/test split for ONE model, not K folds.
    split_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + 1024)
    train_idx, eval_idx = next(iter(split_kfold.split(np.arange(len(dataset)))))
    train_idx = train_idx.tolist()
    eval_idx = eval_idx.tolist()

    dump_folder = None
    if log_folder is not None:
        dump_folder = Path(log_folder)
        dump_folder.mkdir(parents=True, exist_ok=True)

    trainer = _create_trainer(data_configs, train_params, random_seed, dump_folder, dataset_key)
    train_loader, val_loader = _build_loaders_for_fold(
        dataset=dataset,
        indices=train_idx,
        batch_size=batch_size,
        seed=random_seed + 1,
    )
    model, train_result = trainer.train(train_loader, val_loader)
    print(f"[Standalone] train result: stop_reason={train_result.stop_reason}, "
          f"best_epoch={train_result.best_epoch}, best_val_score={train_result.best_val_score}")

    eval_view = _build_dataset_view(dataset, eval_idx)
    effect_estimate = trainer.predict(model, eval_view)
    test_error, n_eval = _compute_unified_h_score(trainer, model, dataset, eval_idx)
    total_runtime_seconds = time.time() - start_time

    if save_model:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / f"{dataset_key}_s{scenario}_standalone_h_seed{random_seed}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"[Standalone] 模型权重已保存至 {model_path}")

    empty_device_cache(trainer.device)

    print(f"\n--- [Standalone] 非 cross-fitting ATE(POR) (Seed {random_seed}) ---")
    print(f"POR = {effect_estimate.item():.4f} | test_error(moment_risk) = {test_error:.6f} | "
          f"total_runtime = {total_runtime_seconds:.2f}s")
    print("----------------------------------")

    return {
        "POR": effect_estimate.item(),
        "test_error": test_error,
        "n_test": n_eval,
        "total_runtime_seconds": total_runtime_seconds,
        "best_epoch": train_result.best_epoch,
        "stopped_early": train_result.stopped_early,
        "stop_reason": train_result.stop_reason,
        "best_state_restored": train_result.best_state_restored,
    }
