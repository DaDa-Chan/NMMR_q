"""
Pure, Dataset/Torch-free effect and bridge-error functions
(plan/p0_e1_e2/01_shared_protocol.md section 6, plan/实验计划.md section 2
and 12.2). These take plain numpy arrays only, so they can be unit-tested
against hand-computed examples without touching any model or dataset code,
and reused identically by P0/E1/E2.
"""
from typing import Any, Dict

import numpy as np


def compute_effect_scores(
    y: np.ndarray,
    a: np.ndarray,
    q_observed: np.ndarray,
    h_observed: np.ndarray,
    h1: np.ndarray,
    h0: np.ndarray,
) -> Dict[str, Any]:
    """
    Per-sample POR/PIPW/PDR scores (plan/实验计划.md section 2) plus their
    sample means. `q_observed`/`h_observed` are each sample's OOF bridge
    prediction at its OWN observed treatment level; `h1`/`h0` are the same
    sample's OOF outcome-bridge prediction forced to A=1/A=0 respectively.

    The final effect estimate is the mean of ALL samples' scores -- never a
    simple average of per-fold estimates (plan/实验计划.md section 3.3: "最终
    效应按所有样本的 score 加权平均；折大小不等时不能简单等权平均各折效应").
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    q_observed = np.asarray(q_observed, dtype=float).reshape(-1)
    h_observed = np.asarray(h_observed, dtype=float).reshape(-1)
    h1 = np.asarray(h1, dtype=float).reshape(-1)
    h0 = np.asarray(h0, dtype=float).reshape(-1)

    # Minimal input checks (Phase 1.1 item 6) -- not a general validation
    # framework, just the three things that would otherwise silently
    # produce a meaningless or misleading score for this SGD protocol.
    lengths = {
        "y": y.size, "a": a.size, "q_observed": q_observed.size,
        "h_observed": h_observed.size, "h1": h1.size, "h0": h0.size,
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"compute_effect_scores: all inputs must have the same length; got {lengths}.")
    n = y.size
    if n == 0:
        raise ValueError("compute_effect_scores: inputs are empty (n=0).")
    valid_a = np.isin(a, [0.0, 1.0])
    if not valid_a.all():
        bad = sorted(set(a[~valid_a].tolist()))
        raise ValueError(f"compute_effect_scores: `a` must only contain 0/1; found other values: {bad}.")

    weight = 2.0 * a - 1.0
    por_scores = h1 - h0
    pipw_scores = weight * q_observed * y
    pdr_scores = weight * q_observed * (y - h_observed) + (h1 - h0)

    return {
        "por_scores": por_scores,
        "pipw_scores": pipw_scores,
        "pdr_scores": pdr_scores,
        "tau_por": float(np.mean(por_scores)),
        "tau_pipw": float(np.mean(pipw_scores)),
        "tau_pdr": float(np.mean(pdr_scores)),
    }


def compute_bridge_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """MSE_{H,a}/MSE_{Q,a} (plan/实验计划.md section 12.2): mean squared
    error between a bridge's prediction and its analytical true value, both
    evaluated at the SAME forced treatment level, on an independent test set."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    return float(np.mean((pred - true) ** 2))


def compute_bridge_mae(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    return float(np.mean(np.abs(pred - true)))


def compute_q_distribution_diagnostics(q_values: np.ndarray) -> Dict[str, float]:
    """Minimal Q diagnostics (plan/p0_e1_e2/05_reporting.md section 1):
    min/p50/p90/p95/p99/max and the fraction of non-positive predictions.
    Reported for context only -- a negative or extreme Q is never itself
    treated as a failure (plan/p0_e1_e2/05_reporting.md section 2)."""
    q_values = np.asarray(q_values, dtype=float).reshape(-1)
    if q_values.size == 0:
        return {
            "q_min": float("nan"), "q_p50": float("nan"), "q_p90": float("nan"),
            "q_p95": float("nan"), "q_p99": float("nan"), "q_max": float("nan"),
            "q_negative_rate": float("nan"),
        }
    return {
        "q_min": float(np.min(q_values)),
        "q_p50": float(np.percentile(q_values, 50)),
        "q_p90": float(np.percentile(q_values, 90)),
        "q_p95": float(np.percentile(q_values, 95)),
        "q_p99": float(np.percentile(q_values, 99)),
        "q_max": float(np.max(q_values)),
        "q_negative_rate": float(np.mean(q_values <= 0.0)),
    }
