"""
SGD scenario registry (plan/p0_e1_e2/01_shared_protocol.md section 3,
plan/p0_e1_e2/plan.md section 4, plan/实验计划.md section 6).

Only S1 (original W, original Z) is enabled this round. S2-S4 are
registered as explicit, named placeholders -- `enabled=False` -- so that
requesting them raises a clear error instead of silently doing nothing or
falling back to anything else. They must NOT be mapped onto
src/data/ate/sgd_pv.py's existing, unrelated `scenario=2/3/4` absolute-value
transforms: those are a different, older numbering and this plan explicitly
forbids reusing them as S2-S4's real transform
(plan/p0_e1_e2/plan.md: "不得把旧生成器里同名的 scenario 2-4 自动当作新 E2 场景").

S2-S4's real invertible transforms, true bridges and true ATE are not
decided yet (plan/实验计划.md section 6: "具体变换形式、真桥与数值规则冻结后
才能启用") -- enabling them is future work, not something this module
guesses at.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.data.ate.sgd_pv import generate_data
from src.models.hpo_protocol import _true_h0_scenario1, _true_q0_scenario1

# plan/实验计划.md section 4.1: this basic DGP's true ATE is 2 for every
# contrast (A=1 vs A=0), independent of sample size or seed.
S1_TRUE_ATE = 2.0


@dataclass(frozen=True)
class ScenarioSpec:
    """
    Minimal, uniform scenario interface (plan/p0_e1_e2/01_shared_protocol.md
    section 3): `generate_train`/`generate_test` take (n, seed) and return a
    pandas DataFrame with the SGDDataset columns (X1, X2, A, Z, W, U, Y) --
    wrapping it into an SGDDataset/hashing it is the caller's job, not this
    registry's, so this module stays free of a torch/device dependency.
    `true_h(w, a, x)`/`true_q(z, a, x)` take numpy arrays (any constant `a`
    level broadcastable against `w`/`z`) and return the analytical bridge
    value. Both are `None` for a disabled scenario.
    """
    scenario_id: str
    enabled: bool
    generate_train: Optional[Callable[[int, int], pd.DataFrame]]
    generate_test: Optional[Callable[[int, int], pd.DataFrame]]
    true_h: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
    true_q: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
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


def _make_disabled_scenario_error(scenario_id: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Scenario {scenario_id} is registered but not enabled "
            "(plan/p0_e1_e2/plan.md section 4: its transform/true bridge/true ATE "
            "have not been frozen yet). This is NOT src/data/ate/sgd_pv.py's old "
            f"scenario={scenario_id[-1]} generator -- reusing that would silently "
            "misrepresent an unfrozen scenario as real, which this registry refuses to do."
        )
    return _raise


def _disabled_scenario(scenario_id: str, representation_metadata: Dict[str, Any]) -> ScenarioSpec:
    raiser = _make_disabled_scenario_error(scenario_id)
    return ScenarioSpec(
        scenario_id=scenario_id, enabled=False,
        generate_train=raiser, generate_test=raiser, true_h=raiser, true_q=raiser,
        true_ate=None, representation_metadata=representation_metadata,
    )


SCENARIOS: Dict[str, ScenarioSpec] = {
    "S1": ScenarioSpec(
        scenario_id="S1", enabled=True,
        generate_train=_s1_generate, generate_test=_s1_generate,
        true_h=_s1_true_h, true_q=_s1_true_q, true_ate=S1_TRUE_ATE,
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
            "plan/p0_e1_e2/plan.md section 4 requires freezing its transform/true bridge/"
            "true ATE before it can be requested."
        )
    return spec
