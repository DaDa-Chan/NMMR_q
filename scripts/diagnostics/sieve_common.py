"""
Shared pieces of the linear-sieve calculations of the shifted batch-V objective
J_tau (paper: Proposition "Linear sieve" and Appendix "The sieve calculation in
the simulation design"). No neural network is trained.

Sieves (correctly specified for scenario S1):
  outcome bridge h(a, .): one coefficient vector per arm on the polynomials of
      degree <= deg in (W, X1, X2), including the constant;
  treatment bridge q_a: the polynomials of degree <= deg in (Z, X1, X2) plus the
      true q_{0,a} itself as one extra (last) basis function, so that q_{0,a}
      lies in the span. This is a device to satisfy the correct-specification
      assumption, not an estimator.

Kernels: Gaussian, k(s, s') = exp(-||s - s'||^2 / (2 * med^2 * mult)), with med
the median pairwise distance among the first 2000 sample points (+1e-5) and
mult the bandwidth multiplier of the frozen configurations (Q: 0.5, H: 1.0).
The diagonal of each kernel matrix is set to zero, so quadratic forms divided
by N(N-1) are full-sample U-statistics.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.scenarios import get_scenario  # noqa: E402

SPEC = get_scenario("S1")
MULT = {"Q": 0.5, "H": 1.0}  # configs/sgd_frozen/{q,h}_v_frozen_config.json
N_TEST = 200_000
TRAIN_SEED0 = 900    # sample used to estimate the matrices: seed TRAIN_SEED0 + s
TEST_SEED0 = 9900    # independent evaluation sample: seed TEST_SEED0 + s
MEDIAN_SUBSAMPLE = 2000


def poly(cols, deg):
    out = [np.ones(len(cols[0]))]
    for d in range(1, deg + 1):
        for c in itertools.combinations_with_replacement(range(len(cols)), d):
            out.append(np.prod([cols[i] for i in c], axis=0))
    return np.c_[tuple(out)]


def x_matrix(df):
    return df[["X1", "X2"]].values


def true_q(df, a):
    return SPEC.true_q(df.Z.values, np.full(len(df), a), x_matrix(df))


def true_h(df, a):
    return SPEC.true_h(df.W.values, np.full(len(df), a), x_matrix(df))


def basis_q(df, a, deg):
    return np.c_[poly([df.Z.values, df.X1.values, df.X2.values], deg), true_q(df, a)]


def basis_h(df, deg):
    return poly([df.W.values, df.X1.values, df.X2.values], deg)


def offdiag_kernel(M, mult):
    sq = (M ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None] - 2 * M @ M.T, 0)
    m = min(MEDIAN_SUBSAMPLE, len(M))
    med = np.median(np.sqrt(D[:m, :m][np.triu_indices(m, 1)])) + 1e-5
    K = np.exp(-D / (2 * med ** 2 * mult))
    np.fill_diagonal(K, 0)
    return K


def draw(n_train, seed, n_test=N_TEST):
    return (SPEC.generate_train(n_train, TRAIN_SEED0 + seed),
            SPEC.generate_test(n_test, TEST_SEED0 + seed))


def outcome_system(df, deg):
    """Matrices of the outcome-bridge objective on the two-arm sieve."""
    N = len(df)
    nn = N * (N - 1)
    A = df.A.values.astype(float)
    KT = offdiag_kernel(df[["A", "Z", "X1", "X2"]].values, MULT["H"])
    P = basis_h(df, deg)
    BH = np.c_[(1 - A)[:, None] * P, A[:, None] * P]
    Y = df.Y.values
    return dict(M=BH.T @ KT @ BH / nn, G=BH.T @ BH / N, m=BH.T @ KT @ Y / nn, g=BH.T @ Y / N, p=P.shape[1])


def treatment_system(df, a, deg):
    """Matrices of the arm-a treatment-bridge objective."""
    N = len(df)
    nn = N * (N - 1)
    KS = offdiag_kernel(df[["W", "X1", "X2"]].values, MULT["Q"])
    I = (df.A.values == a).astype(float)
    C = I[:, None] * basis_q(df, a, deg)
    ones = np.ones(N)
    return dict(M=C.T @ KS @ C / nn, G=C.T @ C / N, m=C.T @ KS @ ones / nn, g=C.T @ ones / N)
