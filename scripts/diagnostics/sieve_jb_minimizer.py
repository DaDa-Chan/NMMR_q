"""
Minimizers of the shifted batch-V objective J_tau on a correctly specified
linear sieve in scenario S1, and the biases they imply (paper: Remark "What is
and is not claimed" and Appendix "The sieve calculation in the simulation
design"). No neural network is trained; there are no tuning parameters.

  Q: J_tau(q) = R_q(q) + tau * E[(1(A=a) q - 1)^2]   kernel on (W, X)
  H: J_tau(h) = R_h(h) + tau * E[(Y - h)^2]          kernel on (A, Z, X)
  tau = 1 / (b - 1)

For each seed, the matrices are estimated on N training points (U-statistics),
the normal equations are solved for every batch size b, and remainders and
biases are evaluated on an independent sample of 200,000 points.

    python scripts/diagnostics/sieve_jb_minimizer.py --n 6000 --seeds 4 --deg 3 \
        --out results/diagnostics/sieve_jb_minimizer_deg3.csv

Ported from plan/q_diagnostics/jb_minimizer.py; the computation is unchanged.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sieve_common as sc

BATCH_SIZES = [16, 32, 64, 128, 256, 500, 1000, 4000, 10 ** 6]


def run(n, seeds, deg, batch_sizes=BATCH_SIZES, n_test=sc.N_TEST):
    rows = []
    for seed in range(seeds):
        df, te = sc.draw(n, seed, n_test)
        At = te.A.values.astype(float)
        H = sc.outcome_system(df, deg)
        Pt = sc.basis_h(te, deg)
        p = H["p"]
        Q = {a: sc.treatment_system(df, a, deg) for a in (0, 1)}
        for b in batch_sizes:
            tau = 1.0 / (b - 1)
            eta = np.linalg.solve(H["M"] + tau * H["G"], H["m"] + tau * H["g"])
            hb = {0: Pt @ eta[:p], 1: Pt @ eta[p:]}
            qb = {}
            r = dict(seed=seed, b=b)
            for a in (0, 1):
                S = Q[a]
                qb[a] = sc.basis_q(te, a, deg) @ np.linalg.solve(S["M"] + tau * S["G"], S["m"] + tau * S["g"])
                I = (At == a).astype(float)
                h0, q0 = sc.true_h(te, a), sc.true_q(te, a)
                dh, dq = hb[a] - h0, qb[a] - q0
                r[f"rem{a}"] = -np.mean(I * dq * dh)
                r[f"dh{a}"] = np.sqrt(np.mean(dh ** 2))
                r[f"dq{a}"] = np.sqrt(np.mean(I * dq ** 2))
                r[f"q{a}_meanerr"] = dq.mean()
                r[f"q{a}_sdratio"] = qb[a].std() / q0.std()
                r[f"h{a}_slopeW"] = np.polyfit(te.W.values, hb[a], 1)[0]
            r["rem_ate"] = r["rem1"] - r["rem0"]
            ate = sc.SPEC.true_ate
            r["por_bias"] = np.mean(hb[1] - hb[0]) - ate
            sgn = 2 * At - 1
            qobs = np.where(At > 0.5, qb[1], qb[0])
            hobs = np.where(At > 0.5, hb[1], hb[0])
            r["pipw_bias"] = np.mean(sgn * qobs * te.Y.values) - ate
            r["pdr_bias"] = np.mean(hb[1] - hb[0] + sgn * qobs * (te.Y.values - hobs)) - ate
            rows.append(r)
        print("seed", seed, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = run(args.n, args.seeds, args.deg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    g = d.groupby("b")
    print(g.mean().drop(columns="seed").round(4).to_string())
    print("\nSE:")
    print(g.sem()[["rem0", "rem1", "rem_ate", "por_bias", "pipw_bias", "pdr_bias"]].round(4).to_string())


if __name__ == "__main__":
    main()
