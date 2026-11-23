# Diagnostics behind numbers quoted in the paper

These scripts do not train neural networks. They compute population quantities on a correctly specified linear sieve in scenario S1, with the data generator and the true bridges taken from `src/experiments/scenarios.py`. Run them from the repository root with the project environment.

| Script | What it reproduces | Runtime (laptop CPU) |
|---|---|---|
| `sieve_jb_minimizer.py` | Appendix "The sieve calculation in the simulation design": ATE remainder and POR bias of the minimizers of the shifted objective for $b=16,\dots,1000$, and the local slopes 0.25–1.03 quoted in the remark on what is and is not claimed | about 15 s per degree |
| `sieve_jb_pairing.py` | Stability checks in the same appendix: spread across seeds at $b=64$ and $b=500$, and the effect of halving $N$; also the weighted spectral pairing decomposition (theory note Section 12) | about 10 s per call |
| `sieve_common.py` | Shared bases, kernels and data draws (not run directly) | — |

```bash
python scripts/diagnostics/sieve_jb_minimizer.py --n 6000 --seeds 4 --deg 3 --out results/diagnostics/sieve_jb_minimizer_deg3.csv
python scripts/diagnostics/sieve_jb_pairing.py --n 6000 --seeds 4 --deg 3 --b 64
python scripts/diagnostics/sieve_jb_pairing.py --n 3000 --seeds 4 --deg 3 --b 64
python scripts/diagnostics/sieve_jb_pairing.py --n 6000 --seeds 4 --deg 3 --b 500
```

Expected output, for checking a rerun:

- **Minimizer, degree 3, 4-seed means.** ATE remainder 0.166, 0.139, 0.102, 0.064, 0.034, 0.017, 0.010 and POR bias 0.244, 0.238, 0.210, 0.165, 0.116, 0.078, 0.050, for $b=16,32,64,128,256,500,1000$. Degrees 2 and 4 agree to the third decimal for $b\le128$.
- **Pairing, ATE remainder of the sample minimizer.**
  - $N=6000$, $b=64$: mean 0.102, SD 0.007 (per seed 0.100, 0.111, 0.104, 0.095).
  - $N=3000$, $b=64$: mean 0.103, SD 0.012.
  - $N=6000$, $b=500$: mean 0.017, SD 0.013.

The scripts were ported from development versions on 2026-11-24. Rerunning them reproduced the saved development outputs exactly: the maximum absolute difference over all columns was 0 for degrees 2, 3 and 4. They use their own fixed seeds: 900+s for the estimation sample and 9900+s for the 200,000-point evaluation sample. These are independent of the pilot data used for tuning and of the confirmatory seed 20261201.

The calculation was carried out after the corresponding NMMR-V results had been seen; it is a post hoc comparison, not a pre-specified prediction.
