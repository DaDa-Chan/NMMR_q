"""
Pilot: does a fixed small batch (and lr) train the NMMR bridges (H and/or Q)
well across sample sizes?

Uses its OWN data seeds (never the P0 repetitions' seeds), the real
cross-fitting path (`_run_cross_fitting`) with a stratified 5-fold plan, and the
frozen H-U/H-V/Q-U/Q-V configs with selected training fields overridden. Rows are
appended to the CSV after every (n, rep, method, cfg) so an interrupted run
resumes by skipping finished keys (an existing CSV without a `bridge`
column is treated as H-only, so older H results remain resumable).

Example:
    python scripts/pilot_bridge_batch.py --ns 500 1000 2000 5000 --reps 5 \
        --out pilot_bridge_batch.csv
    python scripts/pilot_bridge_batch.py --bridges q --ns 500 --reps 1 --out pilot_q.csv
    python scripts/pilot_bridge_batch.py --ns 500 --reps 1 --methods u \
        --configs frozen bs32 "bs48_lr1e-3:batch_size=48,learning_rate=1e-3"
"""
import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ate.data_class import SGDDataset  # noqa: E402
from src.data.ate.sgd_pv import generate_data  # noqa: E402
from src.experiments.scenarios import get_scenario  # noqa: E402
from src.experiments.sgd_experiment_protocol import _derive_bridge_init_seed, build_fold_plan  # noqa: E402
from src.models.hpo_protocol import FrozenConfig  # noqa: E402
from src.models.NMMRh.nmmr_h_experiments import _run_cross_fitting as _run_cf_h  # noqa: E402
from src.models.NMMRq.nmmr_q_experiments import _run_cross_fitting as _run_cf_q  # noqa: E402

PILOT_BASE_SEED = 987654321  # deliberately != the P0 run's base_seed (20261103)

DEFAULT_CONFIGS = {
    "frozen": {},
    "bs32": dict(batch_size=32),
    "bs64": dict(batch_size=64),
    "bs128": dict(batch_size=128),
    "bs256": dict(batch_size=256),
    "bs32_lr2e-3": dict(batch_size=32, learning_rate=2e-3),
    "bs64_lr2e-3": dict(batch_size=64, learning_rate=2e-3),
    "bs128_lr2e-3": dict(batch_size=128, learning_rate=2e-3),
}
INT_KEYS = {"batch_size", "n_epochs", "patience", "network_depth", "network_width"}


def parse_config(spec):
    """'name' (must be a default) or 'name:key=val,key=val'."""
    if ":" not in spec:
        if spec not in DEFAULT_CONFIGS:
            raise SystemExit(f"Unknown config {spec!r}; known: {sorted(DEFAULT_CONFIGS)}")
        return spec, DEFAULT_CONFIGS[spec]
    name, rest = spec.split(":", 1)
    over = {}
    for kv in rest.split(","):
        k, v = kv.split("=")
        over[k] = int(v) if k in INT_KEYS else float(v)
    return name, over


def seeds_for(n, rep):
    ss = np.random.SeedSequence([PILOT_BASE_SEED, n, rep])
    d, t, s, i = (int(x) for x in ss.generate_state(4, dtype=np.uint32))
    return d, t, s, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--bridges", nargs="+", default=["h"], choices=["h", "q"])
    ap.add_argument("--methods", nargs="+", default=["u", "v"], choices=["u", "v"])
    ap.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    ap.add_argument("--out", default="pilot_bridge_batch.csv")
    ap.add_argument("--n-test", type=int, default=20000)
    args = ap.parse_args()

    cfgs = [parse_config(c) for c in args.configs]
    out = Path(args.out)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        if "bridge" not in old.columns:
            old["bridge"] = "h"
        done = set(zip(old.bridge, old.n, old.rep, old.method, old.cfg))
    scen = get_scenario("S1")

    for n in args.ns:
        for rep in range(1, args.reps + 1):
            todo = [(b, m, c) for b in args.bridges for m in args.methods for c in cfgs
                    if (b, n, rep, m, c[0]) not in done]
            if not todo:
                continue
            d_seed, t_seed, s_seed, i_seed = seeds_for(n, rep)
            train = SGDDataset("", df=generate_data(n_samples=n, scenario=1, seed=d_seed), device="cpu")
            test_df = generate_data(n_samples=args.n_test, scenario=1, seed=t_seed)
            test = SGDDataset("", df=test_df, device="cpu")
            w, x = test_df["W"].to_numpy(), test_df[["X1", "X2"]].to_numpy()
            h0_true = scen.true_h(w, np.zeros(len(w)), x)
            h1_true = scen.true_h(w, np.ones(len(w)), x)
            z, a_obs, y_obs = test_df["Z"].to_numpy(), test_df["A"].to_numpy(), test_df["Y"].to_numpy()
            q0_true = scen.true_q(z, np.zeros(len(z)), x)
            q1_true = scen.true_q(z, np.ones(len(z)), x)
            plan = build_fold_plan(train, 5, s_seed)
            for bridge, method, (name, over) in todo:
                fc = FrozenConfig.load(ROOT / f"configs/sgd_frozen/{bridge}_{method}_frozen_config.json")
                tp = fc.to_train_params(bridge.upper())
                tp.update(over)
                run_cf = _run_cf_h if bridge == "h" else _run_cf_q
                t0 = time.monotonic()
                with contextlib.redirect_stdout(io.StringIO()):
                    _, _, _, fd, pred = run_cf(
                        data_name="sgd", dataset=train, n_splits=5, train_params=tp, data_configs={},
                        random_seed=s_seed, log_folder=None, tag="pilot", fold_plan=plan.splits,
                        init_seed=_derive_bridge_init_seed(i_seed, f"nmmr_{method}", bridge.upper()),
                        test_dataset=test,
                    )
                row = dict(
                    bridge=bridge, n=n, rep=rep, method=method, cfg=name,
                    batch_size=tp["batch_size"], lr=tp["learning_rate"],
                )
                if bridge == "h":
                    h0, h1 = pred["h0_test"], pred["h1_test"]
                    row.update(
                        mse=float(np.mean(((h0 - h0_true) ** 2 + (h1 - h1_true) ** 2) / 2)),
                        ate_est=float(np.mean(h1 - h0)),  # POR
                        w_slope=float(np.polyfit(w, h0, 1)[0]),
                    )
                else:
                    q0, q1 = pred["q0_test"], pred["q1_test"]
                    q_obs = np.where(a_obs > 0.5, q1, q0)
                    row.update(
                        mse=float(np.mean(((q0 - q0_true) ** 2 + (q1 - q1_true) ** 2) / 2)),
                        ate_est=float(np.mean((2 * a_obs - 1) * q_obs * y_obs)),  # PIPW on the test set
                        w_slope=np.nan,
                    )
                row.update(
                    max_ep_frac=float(np.mean([d.get("stop_reason") == "max_epochs" for d in fd])),
                    seconds=time.monotonic() - t0,
                )
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(row, flush=True)


if __name__ == "__main__":
    main()
