"""
Confirmatory analysis for the v2 experiments (E1-E4 on S1).

Reads one or more finished SGD protocol result directories, merges them by
(scenario, n, repetition, method, variant), and evaluates the pre-registered
decision rules. Everything here is read-only: it
never trains a model and never writes into a result directory.

Conventions
-----------
* Bias of an estimator in a cell = mean over repetitions of (tau_hat - true_ate);
  its Monte Carlo SE = SD / sqrt(R).
* Intervals in the decision rules are two-sided 90% intervals (z = 1.645).
* Different sample sizes use independent data, so differences across n use
  independent-sample SEs. Different methods at the same (n, repetition) share
  the same data (same base_seed), so method comparisons are paired.
* Slopes (log|bias| on log n, or on log tau for E3) use a percentile bootstrap
  over repetitions within each cell, with a fixed seed. E3 slopes are reported
  as exploratory output only; the E3 decisions are directional (monotone
  decrease in b on a fixed grid, and PDR less biased than POR/PIPW).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Z90 = 1.6448536269514722
ESTIMATOR_COLUMNS: Dict[str, str] = {"POR": "tau_por", "PIPW": "tau_pipw", "PDR": "tau_pdr"}
KEY_COLUMNS: List[str] = ["scenario", "n", "repetition", "method", "variant"]
SUPPORT, NOT_SUPPORT, INSUFFICIENT, MISSING = "支持", "不支持", "证据不足", "缺少数据"


# ---------------------------------------------------------------------------
# Loading and merging
# ---------------------------------------------------------------------------

def _read_dir(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_dir = Path(results_dir)
    effect = pd.read_csv(results_dir / "e2" / "effect_results.csv")
    bridge_path = results_dir / "e1" / "bridge_results.csv"
    bridge = pd.read_csv(bridge_path) if bridge_path.exists() else pd.DataFrame()
    reps = pd.read_csv(results_dir / "common" / "repetitions.csv")
    for df in (effect, bridge, reps):
        if not df.empty:
            df["source_dir"] = str(results_dir)
    for df in (effect, bridge):
        if not df.empty:
            df["variant"] = df["variant"].fillna("").astype(str)
    return effect, bridge, reps


def load_results(results_dirs: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Concatenate effect/bridge/repetition tables of several result directories.

    Raises ValueError if the same (scenario, n, repetition, method, variant)
    appears in two directories, or if the same (scenario, n, repetition) was
    generated with different data in two directories (different data_hash),
    which would break the paired comparisons.
    """
    if not results_dirs:
        raise ValueError("load_results: no result directories given.")
    parts = [_read_dir(Path(d)) for d in results_dirs]
    effect = pd.concat([p[0] for p in parts], ignore_index=True)
    bridge = pd.concat([p[1] for p in parts if not p[1].empty], ignore_index=True) if any(not p[1].empty for p in parts) else pd.DataFrame()
    reps = pd.concat([p[2] for p in parts], ignore_index=True)

    dup = effect.duplicated(subset=KEY_COLUMNS, keep=False)
    if dup.any():
        rows = effect.loc[dup, KEY_COLUMNS + ["source_dir"]].head(10).to_dict("records")
        raise ValueError(f"load_results: duplicated result keys across directories, e.g. {rows}")

    hashes = reps.groupby(["scenario", "n", "repetition"])["data_hash"].nunique()
    bad = hashes[hashes > 1]
    if len(bad):
        raise ValueError(
            "load_results: the same (scenario, n, repetition) has different data in different "
            f"directories (different base_seed?): {list(bad.index[:10])}"
        )
    return effect, bridge, reps


# ---------------------------------------------------------------------------
# Cell summaries
# ---------------------------------------------------------------------------

def cell_bias_table(effect: pd.DataFrame, true_ate: float) -> pd.DataFrame:
    """Bias, SD, MCSE, RMSE and sqrt(n)*bias per (method, variant, n, estimator)."""
    rows = []
    for (method, variant, n), g in effect.groupby(["method", "variant", "n"]):
        for est, col in ESTIMATOR_COLUMNS.items():
            x = g[col].astype(float).to_numpy()
            x = x[np.isfinite(x)]
            r = x.size
            if r == 0:
                continue
            bias = float(np.mean(x) - true_ate)
            sd = float(np.std(x, ddof=1)) if r > 1 else float("nan")
            mcse = sd / math.sqrt(r) if r > 1 else float("nan")
            rows.append(dict(
                method=method, variant=variant, n=int(n), estimator=est, R=r,
                bias=bias, mcse=mcse, ci90_lower=bias - Z90 * mcse, ci90_upper=bias + Z90 * mcse,
                sd=sd, rmse=float(np.sqrt(np.mean((x - true_ate) ** 2))),
                sqrt_n_bias=math.sqrt(n) * bias, sqrt_n_mcse=math.sqrt(n) * mcse,
            ))
    return pd.DataFrame(rows)


def coverage_table(effect: pd.DataFrame, true_ate: float) -> pd.DataFrame:
    """95% PDR interval coverage per (method, variant, n) with binomial MCSE."""
    need = {"ci_lower_pdr", "ci_upper_pdr", "se_pdr"}
    if not need.issubset(effect.columns):
        return pd.DataFrame(columns=["method", "variant", "n", "R", "coverage", "mcse", "mean_ci_length", "mean_se", "sd", "se_sd_ratio"])
    rows = []
    for (method, variant, n), g in effect.groupby(["method", "variant", "n"]):
        lo, hi = g["ci_lower_pdr"].astype(float).to_numpy(), g["ci_upper_pdr"].astype(float).to_numpy()
        se, tau = g["se_pdr"].astype(float).to_numpy(), g["tau_pdr"].astype(float).to_numpy()
        ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(se)
        r = int(ok.sum())
        if r == 0:
            continue
        p = float(np.mean((lo[ok] <= true_ate) & (true_ate <= hi[ok])))
        sd = float(np.std(tau[ok], ddof=1)) if r > 1 else float("nan")
        rows.append(dict(
            method=method, variant=variant, n=int(n), R=r, coverage=p, mcse=math.sqrt(p * (1 - p) / r),
            mean_ci_length=float(np.mean(hi[ok] - lo[ok])), mean_se=float(np.mean(se[ok])),
            sd=sd, se_sd_ratio=float(np.mean(se[ok])) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan"),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap slope
# ---------------------------------------------------------------------------

def bootstrap_log_slope(
    groups: Mapping[float, np.ndarray], truth: float, n_boot: int = 2000, seed: int = 0, level: float = 0.90,
) -> Dict[str, float]:
    """
    OLS slope of log|mean(x_g) - truth| on log(g) over the groups g, with a
    percentile bootstrap interval that resamples repetitions within each group.
    Returns NaN fields when fewer than two groups or a group's |bias| is 0.
    """
    keys = sorted(groups)
    if len(keys) < 2:
        return dict(slope=float("nan"), lower=float("nan"), upper=float("nan"), n_boot_valid=0)
    logx = np.log(np.asarray(keys, dtype=float))
    data = [np.asarray(groups[k], dtype=float) for k in keys]
    data = [d[np.isfinite(d)] for d in data]
    if any(d.size < 2 for d in data):
        return dict(slope=float("nan"), lower=float("nan"), upper=float("nan"), n_boot_valid=0)

    def slope_of(means: np.ndarray) -> float:
        b = np.abs(means - truth)
        if np.any(b <= 0):
            return float("nan")
        return float(np.polyfit(logx, np.log(b), 1)[0])

    point = slope_of(np.array([d.mean() for d in data]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = slope_of(np.array([d[rng.integers(0, d.size, d.size)].mean() for d in data]))
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return dict(slope=point, lower=float("nan"), upper=float("nan"), n_boot_valid=0)
    a = (1 - level) / 2
    return dict(slope=point, lower=float(np.quantile(boots, a)), upper=float(np.quantile(boots, 1 - a)),
                n_boot_valid=int(boots.size))


def slope_table(
    effect: pd.DataFrame, true_ate: float, x_column: str = "n", n_boot: int = 2000, seed: int = 0,
    transform=None,
) -> pd.DataFrame:
    """Bootstrap slope of log|bias| on log(x) for every (method, variant, estimator)."""
    rows = []
    for (method, variant), g in effect.groupby(["method", "variant"]):
        for est, col in ESTIMATOR_COLUMNS.items():
            groups = {}
            for x, gx in g.groupby(x_column):
                key = float(transform(x)) if transform else float(x)
                groups[key] = gx[col].astype(float).to_numpy()
            res = bootstrap_log_slope(groups, true_ate, n_boot=n_boot, seed=seed)
            rows.append(dict(method=method, variant=variant, estimator=est, n_points=len(groups), **res))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    item: str
    result: str
    detail: Dict[str, float] = field(default_factory=dict)
    rule: str = ""


def _cell(bias_tab: pd.DataFrame, method: str, estimator: str, n: int) -> Optional[pd.Series]:
    m = bias_tab[(bias_tab.method == method) & (bias_tab.variant == "") & (bias_tab.estimator == estimator) & (bias_tab.n == n)]
    return None if m.empty else m.iloc[0]


def _contains(lo: float, hi: float, v: float) -> bool:
    return lo <= v <= hi


def decide_plateau(bias_tab: pd.DataFrame, method: str = "nmmr_v", estimator: str = "PDR") -> Decision:
    """
    E2, NMMR-V plateau. With n1 < n2 the two largest sample sizes:
      支持    the 90% CI of bias(n2) - bias(n1) contains 0 AND both biases' 90% CIs exclude 0;
      不支持  the difference CI excludes 0, |bias(n2)| < |bias(n1)|, and bias(n2)'s CI contains 0
              (the bias falls and vanishes);
      证据不足 otherwise.
    """
    item = f"E2：{method} 的 {estimator} 偏差平台"
    rule = decide_plateau.__doc__.strip()
    ns = sorted(bias_tab[(bias_tab.method == method) & (bias_tab.estimator == estimator)].n.unique())
    if len(ns) < 2:
        return Decision(item, MISSING, rule=rule)
    c1, c2 = _cell(bias_tab, method, estimator, ns[-2]), _cell(bias_tab, method, estimator, ns[-1])
    diff = c2.bias - c1.bias
    se = math.sqrt(c1.mcse ** 2 + c2.mcse ** 2)
    lo, hi = diff - Z90 * se, diff + Z90 * se
    both_nonzero = (not _contains(c1.ci90_lower, c1.ci90_upper, 0.0)) and (not _contains(c2.ci90_lower, c2.ci90_upper, 0.0))
    detail = dict(n1=int(ns[-2]), n2=int(ns[-1]), bias_n1=c1.bias, bias_n2=c2.bias, diff=diff, diff_ci90_lower=lo, diff_ci90_upper=hi)
    if _contains(lo, hi, 0.0) and both_nonzero:
        return Decision(item, SUPPORT, detail, rule)
    if (not _contains(lo, hi, 0.0)) and abs(c2.bias) < abs(c1.bias) and _contains(c2.ci90_lower, c2.ci90_upper, 0.0):
        return Decision(item, NOT_SUPPORT, detail, rule)
    return Decision(item, INSUFFICIENT, detail, rule)


def decide_vanish(bias_tab: pd.DataFrame, method: str = "nmmr_c", estimator: str = "PDR") -> Decision:
    """
    E2, NMMR-C bias vanishes. With n1 < n2 the two largest sample sizes, checked in order:
      支持    bias(n2)'s 90% CI contains 0;
      不支持  the method shows the same plateau as decide_plateau's 支持 condition;
      支持    sqrt(n)|bias| does not increase from n1 to n2 (point estimates);
      证据不足 otherwise.
    """
    item = f"E2：{method} 的 {estimator} 偏差消失"
    rule = decide_vanish.__doc__.strip()
    ns = sorted(bias_tab[(bias_tab.method == method) & (bias_tab.estimator == estimator)].n.unique())
    if len(ns) < 2:
        return Decision(item, MISSING, rule=rule)
    c1, c2 = _cell(bias_tab, method, estimator, ns[-2]), _cell(bias_tab, method, estimator, ns[-1])
    detail = dict(n1=int(ns[-2]), n2=int(ns[-1]), bias_n1=c1.bias, bias_n2=c2.bias,
                  bias_n2_ci90_lower=c2.ci90_lower, bias_n2_ci90_upper=c2.ci90_upper,
                  sqrt_n_abs_bias_n1=abs(c1.sqrt_n_bias), sqrt_n_abs_bias_n2=abs(c2.sqrt_n_bias))
    if _contains(c2.ci90_lower, c2.ci90_upper, 0.0):
        return Decision(item, SUPPORT, detail, rule)
    if decide_plateau(bias_tab, method, estimator).result == SUPPORT:
        return Decision(item, NOT_SUPPORT, detail, rule)
    if abs(c2.sqrt_n_bias) <= abs(c1.sqrt_n_bias):
        return Decision(item, SUPPORT, detail, rule)
    return Decision(item, INSUFFICIENT, detail, rule)


def decide_paired_better(effect: pd.DataFrame, true_ate: float, better: str = "nmmr_c", worse: str = "nmmr_v",
                         estimator: str = "PDR", n: Optional[int] = None) -> Decision:
    """
    E2, at the largest common n: paired difference d_r = tau_better,r - tau_worse,r over
    repetitions present for both methods (same data). mean(d) = bias_better - bias_worse.
      支持    90% CI of mean(d) excludes 0 and |bias_better| < |bias_worse|;
      不支持  90% CI excludes 0 and |bias_better| > |bias_worse|;
      证据不足 otherwise.
    """
    item = f"E2：最大 n 处 {better} 的 {estimator} 偏差小于 {worse}"
    rule = decide_paired_better.__doc__.strip()
    col = ESTIMATOR_COLUMNS[estimator]
    a = effect[(effect.method == better) & (effect.variant == "")]
    b = effect[(effect.method == worse) & (effect.variant == "")]
    common_n = sorted(set(a.n) & set(b.n))
    if not common_n:
        return Decision(item, MISSING, rule=rule)
    n = common_n[-1] if n is None else n
    m = a[a.n == n][["repetition", col]].merge(b[b.n == n][["repetition", col]], on="repetition", suffixes=("_a", "_b"))
    d = (m[f"{col}_a"] - m[f"{col}_b"]).astype(float).to_numpy()
    d = d[np.isfinite(d)]
    if d.size < 2:
        return Decision(item, MISSING, rule=rule)
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))
    lo, hi = mean - Z90 * se, mean + Z90 * se
    bias_a = float(m[f"{col}_a"].mean() - true_ate)
    bias_b = float(m[f"{col}_b"].mean() - true_ate)
    detail = dict(n=int(n), R_paired=int(d.size), bias_better=bias_a, bias_worse=bias_b, diff=mean, diff_ci90_lower=lo, diff_ci90_upper=hi)
    if not _contains(lo, hi, 0.0):
        return Decision(item, SUPPORT if abs(bias_a) < abs(bias_b) else NOT_SUPPORT, detail, rule)
    return Decision(item, INSUFFICIENT, detail, rule)


E3_DECISION_GRID: Tuple[int, ...] = (32, 64, 256)
E3_DOMINANCE_SIZES: Tuple[int, ...] = (64, 256)


def _paired_e3(e3: pd.DataFrame, method: str, sizes: Sequence[int]) -> Optional[pd.DataFrame]:
    """Wide table: one row per repetition present at every batch size, columns (estimator, b)."""
    sub = e3[(e3.method == method) & (e3.variant == "") & (e3.batch_size.isin(list(sizes)))]
    if sorted(sub.batch_size.unique()) != sorted(sizes):
        return None
    wide = sub.pivot_table(index="repetition", columns="batch_size", values=list(ESTIMATOR_COLUMNS.values()))
    return wide.dropna()


def decide_e3_monotone(e3: pd.DataFrame, true_ate: float, method: str = "nmmr_v",
                       grid: Sequence[int] = E3_DECISION_GRID) -> Decision:
    """
    E3 (a): the PDR bias of naive batch-V decreases as the batch size b grows, on the
    pre-specified grid b_1 < b_2 < ... (tau = 1/(b-1) decreases). All batch sizes use the same
    data and initialisation seeds, so differences are paired by repetition:
    d_k = mean_r(tau_PDR(b_{k+1}) - tau_PDR(b_k)), with 90% CIs.
      支持    every adjacent d_k < 0 (point estimates) and the end-to-end difference
              bias(b_last) - bias(b_first) has a 90% CI entirely below 0;
      不支持  some adjacent d_k, or the end-to-end difference, has a 90% CI entirely above 0
              (a significant increase);
      证据不足 otherwise.
    (Bias is positive in this design; the rule is stated for the signed bias.)
    """
    item = "E3(a)：naive batch-V 的 PDR 偏差随 b 增大而下降"
    rule = decide_e3_monotone.__doc__.strip()
    grid = sorted(int(b) for b in grid)
    wide = _paired_e3(e3, method, grid)
    if wide is None or len(wide) < 2:
        return Decision(item, MISSING, rule=rule)
    col = ESTIMATOR_COLUMNS["PDR"]
    detail: Dict[str, float] = {"R_paired": int(len(wide))}
    for b in grid:
        detail[f"bias_b{b}"] = float(wide[(col, b)].mean() - true_ate)

    def paired(b_from: int, b_to: int) -> Tuple[float, float, float]:
        d = (wide[(col, b_to)] - wide[(col, b_from)]).to_numpy()
        m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))
        return m, m - Z90 * se, m + Z90 * se

    adj = [paired(grid[k], grid[k + 1]) for k in range(len(grid) - 1)]
    for k, (m, lo, hi) in enumerate(adj):
        detail[f"d_b{grid[k]}_to_b{grid[k + 1]}"] = m
        detail[f"d_b{grid[k]}_to_b{grid[k + 1]}_ci90_lower"] = lo
        detail[f"d_b{grid[k]}_to_b{grid[k + 1]}_ci90_upper"] = hi
    end_m, end_lo, end_hi = paired(grid[0], grid[-1])
    detail.update(end_to_end=end_m, end_to_end_ci90_lower=end_lo, end_to_end_ci90_upper=end_hi)
    if all(m < 0 for m, _, _ in adj) and end_hi < 0:
        return Decision(item, SUPPORT, detail, rule)
    if any(lo > 0 for _, lo, _ in adj) or end_lo > 0:
        return Decision(item, NOT_SUPPORT, detail, rule)
    return Decision(item, INSUFFICIENT, detail, rule)


def decide_e3_dominance(e3: pd.DataFrame, true_ate: float, method: str = "nmmr_v",
                        sizes: Sequence[int] = E3_DOMINANCE_SIZES, n_boot: int = 2000, seed: int = 0) -> Decision:
    """
    E3 (b): under the target shift, PDR is less biased than both single-bridge estimators.
    For each b in the pre-specified sizes and each comparator in {POR, PIPW}:
    D = |bias_comparator| - |bias_PDR|, with a paired percentile bootstrap over repetitions
    (estimators of one repetition share data and fitted bridges).
      支持    all 90% CIs of D lie entirely above 0;
      不支持  some 90% CI of D lies entirely below 0;
      证据不足 otherwise.
    """
    item = "E3(b)：目标偏移下 PDR 的偏差小于 POR 与 PIPW"
    rule = decide_e3_dominance.__doc__.strip()
    sizes = sorted(int(b) for b in sizes)
    wide = _paired_e3(e3, method, sizes)
    if wide is None or len(wide) < 2:
        return Decision(item, MISSING, rule=rule)
    rng = np.random.default_rng(seed)
    r = len(wide)
    idx = rng.integers(0, r, size=(n_boot, r))
    detail: Dict[str, float] = {"R_paired": int(r)}
    above, below = [], []
    for b in sizes:
        pdr = wide[(ESTIMATOR_COLUMNS["PDR"], b)].to_numpy()
        for comp in ("POR", "PIPW"):
            other = wide[(ESTIMATOR_COLUMNS[comp], b)].to_numpy()
            point = abs(other.mean() - true_ate) - abs(pdr.mean() - true_ate)
            boots = np.abs(other[idx].mean(axis=1) - true_ate) - np.abs(pdr[idx].mean(axis=1) - true_ate)
            lo, hi = float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))
            detail[f"D_{comp}_b{b}"] = float(point)
            detail[f"D_{comp}_b{b}_ci90_lower"] = lo
            detail[f"D_{comp}_b{b}_ci90_upper"] = hi
            above.append(lo > 0)
            below.append(hi < 0)
    if all(above):
        return Decision(item, SUPPORT, detail, rule)
    if any(below):
        return Decision(item, NOT_SUPPORT, detail, rule)
    return Decision(item, INSUFFICIENT, detail, rule)


def decide_coverage_nominal(cov: pd.DataFrame, method: str = "nmmr_c", lo: float = 0.93, hi: float = 0.97,
                            fail_below: float = 0.90) -> Decision:
    """
    E4, nominal coverage of the 95% PDR interval for one method:
      支持    coverage lies in [0.93, 0.97] at every n;
      不支持  coverage is below 0.90 at some n;
      证据不足 otherwise.
    """
    item = f"E4：{method} 的 PDR 区间覆盖率接近 0.95"
    rule = decide_coverage_nominal.__doc__.strip()
    c = cov[(cov.method == method) & (cov.variant == "")].sort_values("n")
    if c.empty:
        return Decision(item, MISSING, rule=rule)
    detail = {f"coverage_n{int(r.n)}": float(r.coverage) for r in c.itertuples()}
    if (c.coverage >= lo).all() and (c.coverage <= hi).all():
        return Decision(item, SUPPORT, detail, rule)
    if (c.coverage < fail_below).any():
        return Decision(item, NOT_SUPPORT, detail, rule)
    return Decision(item, INSUFFICIENT, detail, rule)


def decide_coverage_decline(cov: pd.DataFrame, method: str = "nmmr_v") -> Decision:
    """
    E4, coverage of the naive method falls as n grows. Compare the smallest and the largest n
    (independent data): diff = coverage(n_max) - coverage(n_min), binomial SEs.
      支持    90% CI of diff lies entirely below 0;
      不支持  the point estimate of diff is >= 0;
      证据不足 otherwise.
    """
    item = f"E4：{method} 的覆盖率随 n 增大而下降"
    rule = decide_coverage_decline.__doc__.strip()
    c = cov[(cov.method == method) & (cov.variant == "")].sort_values("n")
    if len(c) < 2:
        return Decision(item, MISSING, rule=rule)
    first, last = c.iloc[0], c.iloc[-1]
    diff = float(last.coverage - first.coverage)
    se = math.sqrt(first.mcse ** 2 + last.mcse ** 2)
    lo, hi = diff - Z90 * se, diff + Z90 * se
    detail = dict(n_min=int(first.n), n_max=int(last.n), coverage_n_min=float(first.coverage),
                  coverage_n_max=float(last.coverage), diff=diff, diff_ci90_lower=lo, diff_ci90_upper=hi)
    if hi < 0:
        return Decision(item, SUPPORT, detail, rule)
    if diff >= 0:
        return Decision(item, NOT_SUPPORT, detail, rule)
    return Decision(item, INSUFFICIENT, detail, rule)


# ---------------------------------------------------------------------------
# E1 mechanism metrics (from saved predictions)
# ---------------------------------------------------------------------------

E1_METHODS: Tuple[str, ...] = ("oracle", "nmmr_c", "nmmr_v", "nmmr_u", "nmmr_u_clip", "nmmr_v_clip")


def e1_mechanism_table(effect: pd.DataFrame, scenario_spec, methods: Iterable[str] = E1_METHODS) -> pd.DataFrame:
    """
    Per (method, n): mean over repetitions of
      q_sd_ratio  SD of predicted q_a / SD of true q_a on the test set (both arms averaged);
      q_max       maximum predicted q on the test set;
      h_w_slope   coefficient of W when regressing predicted h_a on (1, W, X1, X2) on the test set
                  (true bridge: 4; the regression function E[Y|A,W,X]: 2.5), both arms averaged.
    Reads predictions/{scenario}/n{n}/rep{rep:04d}/{method}.npz and
    common/prediction_inputs/{scenario}_n{n}_rep{rep:04d}.npz in each row's source_dir.
    """
    rows = []
    sub = effect[effect.method.isin(list(methods)) & (effect.variant == "")]
    for r in sub.itertuples():
        base = Path(r.source_dir)
        pred_path = base / "predictions" / r.scenario / f"n{int(r.n)}" / f"rep{int(r.repetition):04d}" / f"{r.method}.npz"
        inp_path = base / "common" / "prediction_inputs" / f"{r.scenario}_n{int(r.n)}_rep{int(r.repetition):04d}.npz"
        if not (pred_path.exists() and inp_path.exists()):
            continue
        p, c = np.load(pred_path), np.load(inp_path)
        z, w = c["test_z"].astype(float), c["test_w"].astype(float)
        x = np.column_stack([c["test_x1"], c["test_x2"]]).astype(float)
        design = np.column_stack([np.ones_like(w), w, x])
        ratios, slopes = [], []
        for a in (0, 1):
            qa, ha = p[f"q{a}_test"].astype(float), p[f"h{a}_test"].astype(float)
            q_true = scenario_spec.true_q(z, np.full_like(z, a), x)
            ratios.append(np.std(qa) / np.std(q_true))
            slopes.append(np.linalg.lstsq(design, ha, rcond=None)[0][1])
        rows.append(dict(method=r.method, n=int(r.n), repetition=int(r.repetition),
                         q_sd_ratio=float(np.mean(ratios)),
                         q_max=float(max(p["q0_test"].max(), p["q1_test"].max())),
                         h_w_slope=float(np.mean(slopes))))
    if not rows:
        return pd.DataFrame(columns=["method", "n", "R", "q_sd_ratio", "q_max", "h_w_slope"])
    df = pd.DataFrame(rows)
    out = df.groupby(["method", "n"]).agg(R=("repetition", "size"), q_sd_ratio=("q_sd_ratio", "mean"),
                                          q_max=("q_max", "mean"), h_w_slope=("h_w_slope", "mean")).reset_index()
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_analysis(
    e2_dirs: Sequence[Path], e3_dirs: Mapping[int, Path], out_dir: Path, true_ate: float,
    scenario_spec=None, n_boot: int = 2000, seed: int = 20261201, with_e1: bool = True,
) -> Dict[str, object]:
    """Run every table and decision, write them into out_dir, return the decisions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions: List[Decision] = []

    if e2_dirs:
        effect, _, _ = load_results(e2_dirs)
        bias_tab = cell_bias_table(effect, true_ate)
        cov = coverage_table(effect, true_ate)
        slopes = slope_table(effect, true_ate, "n", n_boot=n_boot, seed=seed)
        bias_tab.to_csv(out_dir / "e2_bias_table.csv", index=False)
        cov.to_csv(out_dir / "e4_coverage.csv", index=False)
        slopes.to_csv(out_dir / "e2_slopes.csv", index=False)
        decisions += [
            decide_plateau(bias_tab), decide_vanish(bias_tab), decide_paired_better(effect, true_ate),
            decide_coverage_nominal(cov), decide_coverage_decline(cov),
        ]
        if with_e1 and scenario_spec is not None:
            e1_mechanism_table(effect, scenario_spec).to_csv(out_dir / "e1_mechanism.csv", index=False)

    if e3_dirs:
        frames = []
        for b, d in sorted(e3_dirs.items()):
            eff, _, _ = load_results([Path(d)])
            eff["batch_size"] = int(b)
            eff["tau"] = 1.0 / (int(b) - 1)
            frames.append(eff)
        e3 = pd.concat(frames, ignore_index=True)
        e3_bias = cell_bias_table(e3.assign(n=e3["batch_size"]), true_ate).rename(columns={"n": "batch_size"})
        e3_bias = e3_bias.drop(columns=["sqrt_n_bias", "sqrt_n_mcse"])  # n is fixed in E3
        e3_bias["tau"] = 1.0 / (e3_bias["batch_size"] - 1)
        e3_slopes = slope_table(e3, true_ate, "tau", n_boot=n_boot, seed=seed)
        e3_bias.to_csv(out_dir / "e3_bias_table.csv", index=False)
        e3_slopes.to_csv(out_dir / "e3_slopes.csv", index=False)
        # Slopes in log tau are exploratory only: the tau^2 order of the PDR bias is
        # asymptotic and is not reached on this batch-size range.
        decisions.append(decide_e3_monotone(e3, true_ate))
        decisions.append(decide_e3_dominance(e3, true_ate, n_boot=n_boot, seed=seed))

    payload = [dict(item=d.item, result=d.result, detail=d.detail, rule=d.rule) for d in decisions]
    (out_dir / "decisions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
    lines = ["# 确认性分析判定", ""]
    for d in decisions:
        lines.append(f"- **{d.item}**：{d.result}")
        for k, v in d.detail.items():
            lines.append(f"  - {k} = {v:.6g}" if isinstance(v, float) else f"  - {k} = {v}")
    (out_dir / "decisions.md").write_text("\n".join(lines) + "\n")
    return {"decisions": payload, "out_dir": str(out_dir)}
