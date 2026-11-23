"""
SGD scenario registry.

Only S1 (original W, original Z) is enabled. S2-S4 are registered as
placeholders with `enabled=False`, so requesting them raises a clear error.
They are unrelated to the `scenario=2/3/4` absolute-value transforms in
src/data/ate/sgd_pv.py, which use an older numbering and must not be reused
here. Their transforms, true bridges and true ATE are not yet fixed.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from src.data.ate.sgd_pv import generate_data
from src.models.hpo_protocol import _true_h0_scenario1, _true_q0_scenario1

# The true ATE of the basic DGP is 2, independent of sample size or seed.
S1_TRUE_ATE = 2.0


@dataclass(frozen=True)
class ScenarioSpec:
    """
    Minimal, uniform scenario interface: `generate_train`/`generate_test`
    take (n, seed) and return a pandas DataFrame with the SGDDataset
    columns (X1, X2, A, Z, W, U, Y) --
    wrapping it into an SGDDataset/hashing it is the caller's job, not this
    registry's, so this module stays free of a torch/device dependency.
    `true_h(w, a, x)`/`true_q(z, a, x)` take numpy arrays (any constant `a`
    level broadcastable against `w`/`z`) and return the analytical bridge
    value. `true_propensity(w, x)` (used by the `oracleP` LS variant)
    returns the analytical P(A=1|W,X).
    All three are `None` for a disabled scenario.
    """
    scenario_id: str
    enabled: bool
    generate_train: Optional[Callable[[int, int], pd.DataFrame]]
    generate_test: Optional[Callable[[int, int], pd.DataFrame]]
    true_h: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
    true_q: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
    true_propensity: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    true_ate: Optional[float]
    representation_metadata: Dict[str, Any] = field(default_factory=dict)


def _s1_generate(n: int, seed: int) -> pd.DataFrame:
    """S1 = the unmodified basic DGP (src/data/ate/sgd_pv.py's scenario=1,
    which is itself a no-op passthrough of the base mechanism)."""
    return generate_data(n_samples=n, scenario=1, seed=seed)


def _s1_true_h(w: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _true_h0_scenario1(w, a, x)


def _s1_true_q(z: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    # hpo_protocol's helper takes (z, x, a); the registry's own convention
    # (matching true_h's (w, a, x)) puts the treatment level second.
    return _true_q0_scenario1(z, x, a)


def _s1_true_propensity(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """pi_0(w,x) = expit(-0.0390625 + 0.125*w - 0.15625*x1 - 0.15625*x2)
    (used by the `oracleP` LS variant)."""
    return expit(-0.0390625 + 0.125 * w - 0.15625 * x[:, 0] - 0.15625 * x[:, 1])


def _make_disabled_scenario_error(scenario_id: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Scenario {scenario_id} is registered but not enabled "
            "(its transform, true bridges and true ATE are not fixed yet). It is "
            f"unrelated to the old scenario={scenario_id[-1]} generator in src/data/ate/sgd_pv.py."
        )
    return _raise


def _disabled_scenario(scenario_id: str, representation_metadata: Dict[str, Any]) -> ScenarioSpec:
    raiser = _make_disabled_scenario_error(scenario_id)
    return ScenarioSpec(
        scenario_id=scenario_id, enabled=False,
        generate_train=raiser, generate_test=raiser, true_h=raiser, true_q=raiser,
        true_propensity=raiser,
        true_ate=None, representation_metadata=representation_metadata,
    )


SCENARIOS: Dict[str, ScenarioSpec] = {
    "S1": ScenarioSpec(
        scenario_id="S1", enabled=True,
        generate_train=_s1_generate, generate_test=_s1_generate,
        true_h=_s1_true_h, true_q=_s1_true_q, true_propensity=_s1_true_propensity,
        true_ate=S1_TRUE_ATE,
        representation_metadata={"w_transform": "identity", "z_transform": "identity"},
    ),
    "S2": _disabled_scenario("S2", {"w_transform": "invertible_nonlinear(frozen pending)", "z_transform": "identity"}),
    "S3": _disabled_scenario("S3", {"w_transform": "identity", "z_transform": "invertible_nonlinear(frozen pending)"}),
    "S4": _disabled_scenario("S4", {"w_transform": "invertible_nonlinear(frozen pending)", "z_transform": "invertible_nonlinear(frozen pending)"}),
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    """
    Look up a scenario by id, raising a clear error for both an unknown id
    and a known-but-disabled one (never silently substituting S1 or an old
    sgd_pv.py scenario number).
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id {scenario_id!r}; known ids: {sorted(SCENARIOS)}.")
    spec = SCENARIOS[scenario_id]
    if not spec.enabled:
        raise ValueError(
            f"Scenario {scenario_id!r} is registered but enabled=False -- "
            "its transform, true bridges and true ATE must be fixed before it can be requested."
        )
    return spec
