"""
Weighted spectral pairing decomposition of the sieve PDR remainder at one batch
size (paper: Proposition "Linear sieve"; theory note Section 12), plus the ATE
remainder of the sample minimizer, used for the stability checks reported in
the paper's appendix (spread across seeds, and halving N).

  Bias_a = -sum_{k,l} a_k d_l (v_k' C u_l) * tau/(tau+nu_k) * tau/(tau+om_l)
  Q side generalized eigenpairs (nu_k, v_k): M_q v = nu G_q v
  H side generalized eigenpairs (om_l, u_l): M_h u = om G_h u
  a_k = v_k' G_q (theta_D - theta_0),  d_l = u_l' G_h (eta_D - eta_0)

Directions are grouped by eigenvalue: 'weak' (> tau), 'mid' (FLOOR..tau),
'unres' (< FLOOR, including negative values, below sampling noise).

    python scripts/diagnostics/sieve_jb_pairing.py --n 6000 --seeds 4 --deg 3 --b 64

Ported from plan/q_diagnostics/jb_pairing.py; the computation is unchanged,
except that the true coefficient vectors are now obtained by least squares on
the sieve and checked to reproduce the true bridges, instead of being typed in.
"""
import argparse

import numpy as np
import scipy.linalg as sl

import sieve_common as sc

FLOOR = 2e-4
NAMES = ["weak", "mid", "unres"]


def true_coefficients(df, deg):
    """eta_0 (outcome, two arms stacked) and theta_0 (treatment, per arm) with exact fit checks."""
    P = sc.basis_h(df, deg)
    p = P.shape[1]
    e0 = np.zeros(2 * p)
    for a in (0, 1):
        coef, *_ = np.linalg.lstsq(P, sc.true_h(df, a), rcond=None)
        assert np.max(np.abs(P @ coef - sc.true_h(df, a))) < 1e-8, "h0 is not in the outcome sieve"
        e0[a * p:(a + 1) * p] = coef
    t0 = {}
    for a in (0, 1):
        k = sc.basis_q(df, a, deg).shape[1]
        t0[a] = np.zeros(k)
        t0[a][-1] = 1.0  # the last basis function is q_{0,a} itself
    return e0, t0


def run(n, seeds, deg, b, n_test=sc.N_TEST):
    tau = 1 / (b - 1)
    sym = lambda M: (M + M.T) / 2  # noqa: E731
    grp = lambda ev: np.where(ev > tau, 0, np.where(ev > FLOOR, 1, 2))  # noqa: E731
    T_sat, T_b, act, chk, cnt = [], [], [], [], []
    for seed in range(seeds):
        df, te = sc.draw(n, seed, n_test)
        At = te.A.values.astype(float)
        H = sc.outcome_system(df, deg)
        p = H["p"]
        HK, HG = sym(H["M"]), sym(H["G"])
        Pt = sc.basis_h(te, deg)
        e0, t0 = true_coefficients(df, deg)
        om, U = sl.eigh(HK, HG)
        d = U.T @ HG @ (np.linalg.solve(HG, H["g"]) - e0)
        gh, fh = grp(om), tau / (tau + om)
        eta = np.linalg.solve(HK + tau * HG, H["m"] + tau * H["g"])
        Ts, Tb = np.zeros((3, 3)), np.zeros((3, 3))
        ra, rs = {}, {}
        for a in (0, 1):
            S = sc.treatment_system(df, a, deg)
            QK, QG = sym(S["M"]), sym(S["G"])
            nu, V = sl.eigh(QK, QG)
            ak = V.T @ QG @ (np.linalg.solve(QG, S["g"]) - t0[a])
            gq, fq = grp(nu), tau / (tau + nu)
            It = (At == a).astype(float)
            Bt = sc.basis_q(te, a, deg)
            chi = np.zeros((len(te), 2 * p))
            chi[:, a * p:(a + 1) * p] = Pt
            C = (It[:, None] * Bt).T @ chi / len(te)
            W1 = -(ak[:, None] * d[None]) * (V.T @ C @ U)
            Wb = W1 * fq[:, None] * fh[None]
            sg = 1 if a == 1 else -1
            for i in range(3):
                for j in range(3):
                    Ts[i, j] += sg * W1[np.ix_(gq == i, gh == j)].sum()
                    Tb[i, j] += sg * Wb[np.ix_(gq == i, gh == j)].sum()
            th = np.linalg.solve(QK + tau * QG, S["m"] + tau * S["g"])
            ra[a] = -np.mean(It * (Bt @ th - Bt[:, -1]) * (chi @ (eta - e0)))
            rs[a] = Wb.sum()
            cnt.append([(gq == i).sum() for i in range(3)])
        T_sat.append(Ts)
        T_b.append(Tb)
        act.append(ra[1] - ra[0])
        chk.append(rs[1] - rs[0])
        cnt.append([(gh == i).sum() for i in range(3)])
    return dict(tau=tau, T_sat=np.array(T_sat), T_b=np.array(T_b), act=np.array(act), chk=np.array(chk),
                cnt=np.array(cnt).reshape(seeds, 3, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--b", type=int, default=64)
    args = ap.parse_args()
    r = run(args.n, args.seeds, args.deg, args.b)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(f"N={args.n} deg={args.deg} b={args.b} tau={r['tau']:.4f}  rows=Q group {NAMES}, cols=H group {NAMES}")
    print("direction counts per seed [Q0,Q1,H]x[weak,mid,unres]:", r["cnt"].tolist())
    a = r["act"]
    print("ATE remainder, actual sample minimizer: mean %.4f  sd across seeds %.4f  per seed %s"
          % (a.mean(), a.std(ddof=1), np.round(a, 4)))
    print("ATE remainder, pairing sum (signal-only): mean %.4f" % r["chk"].mean())
    print("pairing table at this b (ATE, mean over seeds):\n", r["T_b"].mean(0))
    print("SE:\n", r["T_b"].std(0, ddof=1) / np.sqrt(args.seeds))
    print("pairing table with all response factors = 1 (tau->inf limit):\n", r["T_sat"].mean(0),
          " total %.4f" % r["T_sat"].mean(0).sum())


if __name__ == "__main__":
    main()
