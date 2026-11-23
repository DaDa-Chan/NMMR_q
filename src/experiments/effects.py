"""
Effect and bridge-error functions on plain numpy arrays (no Dataset or
torch), shared by P0/E1/E2 and unit-testable against hand-computed examples.
"""
from statistics import NormalDist
from typing import Any, Dict

import numpy as np


def compute_influence_ci(scores: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    """
    Cross-fitted influence-function variance and Wald interval for an effect
    estimate that is the mean of per-sample scores:
        tau_hat = mean(phi_i),  V_hat = mean((phi_i - tau_hat)^2),
        se = sqrt(V_hat / n),   CI = tau_hat -/+ z_{1-(1-level)/2} * se.
    Each phi_i must be the score of sample i under bridges fitted WITHOUT
    sample i (out-of-fold predictions), which is what compute_effect_scores
    receives. Only second moments are used; no fold-wise adjustment is needed.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = scores.size
    if n < 2:
        raise ValueError("compute_influence_ci: need at least 2 scores.")
    if not 0.0 < level < 1.0:
        raise ValueError(f"compute_influence_ci: level must be in (0, 1); got {level}.")
    tau_hat = float(np.mean(scores))
    var_hat = float(np.mean((scores - tau_hat) ** 2))
    se = float(np.sqrt(var_hat / n))
    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    return {
        "tau": tau_hat, "var": var_hat, "se": se,
        "ci_lower": tau_hat - z * se, "ci_upper": tau_hat + z * se,
    }


def compute_effect_scores(
    y: np.ndarray,
    a: np.ndarray,
    q_observed: np.ndarray,
    h_observed: np.ndarray,
    h1: np.ndarray,
    h0: np.ndarray,
) -> Dict[str, Any]:
    """
    Per-sample POR/PIPW/PDR scores plus their
    sample means. `q_observed`/`h_observed` are each sample's OOF bridge
    prediction at its OWN observed treatment level; `h1`/`h0` are the same
    sample's OOF outcome-bridge prediction forced to A=1/A=0 respectively.

    The effect estimate is the mean of ALL samples' scores, not an equal-weight
    average of per-fold estimates (folds may differ in size).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    q_observed = np.asarray(q_observed, dtype=float).reshape(-1)
    h_observed = np.asarray(h_observed, dtype=float).reshape(-1)
    h1 = np.asarray(h1, dtype=float).reshape(-1)
    h0 = np.asarray(h0, dtype=float).reshape(-1)

    # Minimal input checks -- not a general validation
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

    result = {
        "por_scores": por_scores,
        "pipw_scores": pipw_scores,
        "pdr_scores": pdr_scores,
        "tau_por": float(np.mean(por_scores)),
        "tau_pipw": float(np.mean(pipw_scores)),
        "tau_pdr": float(np.mean(pdr_scores)),
    }
    # 95% influence-function interval for PDR only:
    # POR/PIPW have no matching sqrt(n) theory here and are compared as point
    # estimates. Needs n >= 2; with n == 1 the interval fields are NaN.
    if n >= 2:
        ci = compute_influence_ci(pdr_scores, level=0.95)
        result.update(var_pdr=ci["var"], se_pdr=ci["se"],
                      ci_lower_pdr=ci["ci_lower"], ci_upper_pdr=ci["ci_upper"])
    else:
        result.update(var_pdr=float("nan"), se_pdr=float("nan"),
                      ci_lower_pdr=float("nan"), ci_upper_pdr=float("nan"))
    return result


def compute_bridge_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """MSE_{H,a}/MSE_{Q,a}: mean squared
    error between a bridge's prediction and its analytical true value, both
    evaluated at the SAME forced treatment level, on an independent test set."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    return float(np.mean((pred - true) ** 2))


def compute_bridge_mae(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    return float(np.mean(np.abs(pred - true)))


H_MOMENT_FEATURE_NAMES: tuple = ("1", "Z", "A", "X1", "X2")
Q_MOMENT_FEATURE_NAMES: tuple = ("1", "W", "X1", "X2")


def compute_h_test_moments(
    y: np.ndarray, a: np.ndarray, z: np.ndarray, x1: np.ndarray, x2: np.ndarray,
    h0: np.ndarray, h1: np.ndarray,
) -> Dict[str, float]:
    """
    Fixed, finite-dimensional H test moment on the independent test set:
    `h_hat_A = A*h1 + (1-A)*h0` (the H bridge evaluated at each sample's
    OWN observed treatment), `r_H = Y - h_hat_A`, `g_H = (1,Z,A,X1,X2)`
    (`H_MOMENT_FEATURE_NAMES`), `m_H = mean(r_H * g_H)` (elementwise over
    samples, a 5-vector). No new tunable parameter -- never a kernel, never
    a weak-instrument-style threshold; diagnostic only, never a failure gate.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    h0 = np.asarray(h0, dtype=float).reshape(-1)
    h1 = np.asarray(h1, dtype=float).reshape(-1)

    h_a = a * h1 + (1.0 - a) * h0
    r_h = y - h_a
    g_h = np.column_stack([np.ones_like(a), z, a, x1, x2])
    m_h = np.mean(r_h[:, None] * g_h, axis=0)
    return {
        "h_moment_l2": float(np.linalg.norm(m_h, ord=2)),
        "h_moment_linf": float(np.max(np.abs(m_h))),
    }


def compute_q_test_moments(
    a: np.ndarray, w: np.ndarray, x1: np.ndarray, x2: np.ndarray, q0: np.ndarray, q1: np.ndarray,
) -> Dict[str, float]:
    """
    Fixed, finite-dimensional Q test moment: for each forced treatment
    level `a_level` in {0,1}, `r_{Q,a} = I(A==a_level)*q_a - 1`, `g_Q =
    (1,W,X1,X2)` (`Q_MOMENT_FEATURE_NAMES`), `m_{Q,a} = mean(r_{Q,a}*g_Q)`
    (a 4-vector each); `m_Q` concatenates `m_{Q,0}` then `m_{Q,1}` into one
    8-vector. No new tunable parameter; diagnostic only, never a failure gate.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    q1 = np.asarray(q1, dtype=float).reshape(-1)

    g_q = np.column_stack([np.ones_like(a), w, x1, x2])
    moments = []
    for a_level, q_a in ((0.0, q0), (1.0, q1)):
        indicator = (a == a_level).astype(float)
        r_q = indicator * q_a - 1.0
        moments.append(np.mean(r_q[:, None] * g_q, axis=0))
    m_q = np.concatenate(moments)
    return {
        "q_moment_l2": float(np.linalg.norm(m_q, ord=2)),
        "q_moment_linf": float(np.max(np.abs(m_q))),
    }


def compute_q_distribution_diagnostics(q_values: np.ndarray) -> Dict[str, float]:
    """Minimal Q diagnostics: min/p50/p90/p95/p99/max and the fraction of
    non-positive predictions. Context only; a negative or extreme Q is never
    treated as a failure."""
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
