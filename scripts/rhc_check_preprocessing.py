"""
Check the RHC analysis frame against published estimates.

Builds the frame with src/data/ate/rhc_protocol_data.py (full sample, no
splitting) and computes two estimators whose published values depend only
on the data, the covariate coding and the outcome scale:

* the standard doubly robust estimator under no unmeasured confounding
  (logistic propensity and linear outcome regression on L = (X, Z, W)),
  with an influence-function SE;
* the parametric proximal outcome regression (POR) estimator, h linear in
  (A, W, X) with instruments (1, Z, A, X), with a sandwich SE.

Reference values are from Table 3 of Cui et al. (2024), "Semiparametric
proximal causal inference", JASA 119(546). Agreement to two decimals
indicates that the sample, the 67 covariates and the outcome scale match.

Usage:
    python scripts/rhc_check_preprocessing.py
Exit status 1 if either estimate differs from the reference by more than 0.01.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ate.rhc_protocol_data import W_COLUMNS, X_COLUMNS, Z_COLUMNS, build_rhc_frame  # noqa: E402
from src.models.LINEAR.linear_model import ProximalLinearModel  # noqa: E402

# Published reference values: estimate (SD).
REFERENCE = {"DR": (-1.17, 0.32), "POR": (-1.80, 0.44)}
TOLERANCE = 0.01


def standard_dr(df):
    """AIPW under no unmeasured confounding, adjusting for L = (X, Z, W)."""
    L = df[X_COLUMNS + Z_COLUMNS + W_COLUMNS].to_numpy(dtype=float)
    A = df["A"].to_numpy()
    Y = df["Y"].to_numpy()
    L_s = (L - L.mean(0)) / L.std(0)
    ps = LogisticRegression(C=1e8, max_iter=20000).fit(L_s, A).predict_proba(L_s)[:, 1]
    reg = LinearRegression().fit(np.column_stack([A, L]), Y)
    m1 = reg.predict(np.column_stack([np.ones_like(A), L]))
    m0 = reg.predict(np.column_stack([np.zeros_like(A), L]))
    m_obs = np.where(A == 1, m1, m0)
    phi = (2 * A - 1) * (Y - m_obs) / np.where(A == 1, ps, 1 - ps) + m1 - m0
    return float(phi.mean()), float(phi.std(ddof=1) / np.sqrt(len(phi)))


def proximal_or(df):
    """Parametric POR and its sandwich SE (the h block of the stacked estimating equations)."""
    model = ProximalLinearModel(df, z_cols=Z_COLUMNS, w_cols=W_COLUMNS, x_cols=X_COLUMNS)
    b, _ = model.solve_h()
    n = model.n
    M = np.hstack([model.ones, model.Z, model.A.reshape(-1, 1), model.X])
    V = np.hstack([model.ones, model.A.reshape(-1, 1), model.W, model.X])
    rows = model.h_moment_rows(b)
    J_inv = np.linalg.inv(-(M.T @ V) / n)
    cov = J_inv @ (rows.T @ rows / n) @ J_inv.T / n
    # POR = coefficient on A (h is linear in A), the second parameter.
    return float(b[1]), float(np.sqrt(cov[1, 1]))


def main():
    df = build_rhc_frame()
    print(f"n={len(df)}, treated={int(df['A'].sum())}, |Z|={len(Z_COLUMNS)}, "
          f"|W|={len(W_COLUMNS)}, |X|={len(X_COLUMNS)}")
    ours = {"DR": standard_dr(df), "POR": proximal_or(df)}
    ok = True
    print(f"\n{'':16s}{'ours':>18s}{'reference':>18s}")
    for k, (est, se) in ours.items():
        ref_est, ref_se = REFERENCE[k]
        match = abs(est - ref_est) <= TOLERANCE
        ok &= match
        print(f"{k:16s}{est:>10.3f} ({se:.3f}){ref_est:>11.2f} ({ref_se:.2f})   {'ok' if match else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
