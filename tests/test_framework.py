"""Design-invariant tests.

These check the properties the identification strategy actually rests on, not
incidental behaviour. If one of them fails, the attribution table is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.toy2layer import base_scenario, build_case
from coupling0729.core.coupling import PROPAGATION, CouplingEdge, CouplingOperator, build_operator
from coupling0729.core.hazard import DamageState, LognormalFragility
from coupling0729.core.scenario import ARM_NAMES, Arm, CouplingDesign, HazardWindow, Scenario
from coupling0729.experiments import paired_effects, run_factorial
from coupling0729.metrics import MetricConfig, curve_metrics


# --- hazard and fragility --------------------------------------------------
def test_fragility_is_monotone_in_intensity():
    model = LognormalFragility(medians=(0.3, 0.5, 0.75, 1.05), betas=(0.45,) * 4)
    low, high = model.exceedance(0.2), model.exceedance(1.5)
    assert np.all(high >= low - 1e-12)
    assert np.all(np.diff(low) <= 1e-12), "exceedance must be non-increasing in state"


def test_damage_accrues_monotonically_in_time():
    engine = build_case(n_supply=8, n_service=8)
    scenario = base_scenario()
    from coupling0729.core.hazard import sample_damage

    positions, classes = engine._component_inventory()
    damage = sample_damage(
        positions, classes, engine.fragility, engine.hazard_field,
        scenario.window, seed_hazard=3,
    )
    grid = np.linspace(scenario.window.t_oe, scenario.window.t_ee, 25)
    for component in list(positions)[:8]:
        states = [int(damage.state_at(component, t)) for t in grid]
        assert states == sorted(states), "damage must never heal during the hazard"


# --- coupling semantics ----------------------------------------------------
def test_kappa_zero_is_indistinguishable_from_no_edge():
    operator = CouplingOperator(
        phase=PROPAGATION, edges=(CouplingEdge("A:0", "B:0", kappa=0.0),)
    )
    assert operator.support_level("B:0", {"A:0": 0.0}) == pytest.approx(1.0)
    assert operator.is_empty()


def test_kappa_is_the_transmitted_fraction():
    operator = CouplingOperator(
        phase=PROPAGATION, edges=(CouplingEdge("A:0", "B:0", kappa=0.6),)
    )
    # source at 0.25 -> deficit 0.75 -> transmitted 0.45 -> support 0.55
    assert operator.support_level("B:0", {"A:0": 0.25}) == pytest.approx(0.55)


def test_builder_realises_requested_breadth():
    targets = [f"B:{i}" for i in range(20)]
    operator = build_operator(
        PROPAGATION, sources=["A:0", "A:1"], targets=targets,
        q=0.35, kappa=0.5, seed_design=7,
    )
    assert operator.q(targets) == pytest.approx(0.35, abs=0.051)
    assert operator.kappa_bar() == pytest.approx(0.5)


def test_placement_depends_only_on_design_seed():
    targets = [f"B:{i}" for i in range(20)]
    kwargs = dict(sources=["A:0"], targets=targets, q=0.5, kappa=0.5)
    a = build_operator(PROPAGATION, seed_design=11, **kwargs)
    b = build_operator(PROPAGATION, seed_design=11, **kwargs)
    c = build_operator(PROPAGATION, seed_design=12, **kwargs)
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


# --- scenario and arms -----------------------------------------------------
def test_arms_share_a_pairing_key():
    scenario = base_scenario()
    keys = {scenario.for_arm(a).pairing_key() for a in ARM_NAMES}
    assert len(keys) == 1, "arms must differ only in the arm field"


def test_switching_an_arm_off_zeroes_that_phase():
    scenario = base_scenario(q_prop=0.6, kappa_prop=0.8, q_rec=0.6, kappa_rec=0.8)
    off = scenario.for_arm("01").effective_design
    assert off.q_prop == 0.0 and off.kappa_prop == 0.0
    assert off.q_rec == 0.6 and off.kappa_rec == 0.8


def test_window_rejects_zero_width_hazard():
    with pytest.raises(ValueError):
        Scenario(window=HazardWindow(t_oe=10.0, t_ee=10.0, t_mob=5.0))


# --- metrics ---------------------------------------------------------------
def _flat_then_dip(window: HazardWindow, depth: float, recover_at: float) -> pd.Series:
    t = np.arange(window.t_oe, window.t_oe + 500.0 + 1e-9, 1.0)
    f = np.ones_like(t)
    f[(t >= window.t_ee) & (t < recover_at)] = 1.0 - depth
    ramp = (t >= window.t_oe) & (t < window.t_ee)
    f[ramp] = np.linspace(1.0, 1.0 - depth, ramp.sum())
    return pd.Series(f, index=pd.Index(t, name="t"))


def test_lambda_is_read_only_inside_the_embargo():
    window = HazardWindow(t_oe=0.0, t_ee=20.0, t_mob=10.0)
    config = MetricConfig(T_eval=400.0)
    curve = _flat_then_dip(window, depth=0.4, recover_at=200.0)
    assert curve_metrics(curve, window, config)["Lambda"] == pytest.approx(0.4)

    # A deeper dip *after* the embargo must not change Lambda.
    deeper = curve.copy()
    deeper.loc[deeper.index > window.t_repair_start + 20] = 0.1
    assert curve_metrics(deeper, window, config)["Lambda"] == pytest.approx(0.4)


def test_t_rec_is_anchored_at_hazard_end():
    window = HazardWindow(t_oe=0.0, t_ee=20.0, t_mob=10.0)
    config = MetricConfig(T_eval=400.0, eta=0.01, eta_steps=2)
    metrics = curve_metrics(_flat_then_dip(window, 0.4, recover_at=120.0), window, config)
    assert metrics["T_rec"] == pytest.approx(100.0)
    assert not metrics["T_rec_censored"]


def test_censoring_is_flagged_not_silently_dropped():
    window = HazardWindow(t_oe=0.0, t_ee=20.0, t_mob=10.0)
    config = MetricConfig(T_eval=100.0)
    metrics = curve_metrics(_flat_then_dip(window, 0.4, recover_at=900.0), window, config)
    assert metrics["T_rec_censored"] is True
    assert np.isnan(metrics["T_rec"])
    assert metrics["lock_in"] == 1.0


def test_metrics_reject_a_horizon_the_run_cannot_cover():
    window = HazardWindow(t_oe=0.0, t_ee=20.0, t_mob=10.0)
    with pytest.raises(ValueError, match="evaluation horizon"):
        curve_metrics(_flat_then_dip(window, 0.2, 100.0), window, MetricConfig(T_eval=5000.0))


# --- identification --------------------------------------------------------
def test_arms_see_identical_damage():
    engine = build_case(n_supply=10, n_service=10)
    scenario = base_scenario().for_replication(4)
    summaries = [
        engine.run(scenario.for_arm(arm)).damage_summary for arm in ARM_NAMES
    ]
    assert all(s == summaries[0] for s in summaries), "common random numbers broken"


def test_arm_00_has_no_coupling_effect_at_all():
    engine = build_case(n_supply=10, n_service=10)
    high = base_scenario(q_prop=0.9, kappa_prop=1.0, q_rec=0.9, kappa_rec=1.0)
    low = base_scenario(q_prop=0.1, kappa_prop=0.1, q_rec=0.1, kappa_rec=0.1)
    a = engine.run(high.for_arm("00")).operational
    b = engine.run(low.for_arm("00")).operational
    pd.testing.assert_frame_equal(a, b)


def test_additive_identity_holds_exactly():
    engine = build_case(n_supply=10, n_service=10)
    runs, _ = run_factorial(engine, base_scenario(), replications=4)
    effects = paired_effects(runs, metrics=("loss", "Lambda"))
    for metric in ("loss", "Lambda"):
        parts = effects[
            [f"{metric}__delta_prop", f"{metric}__delta_rec", f"{metric}__delta_int"]
        ].sum(axis=1)
        np.testing.assert_allclose(parts, effects[f"{metric}__total"], atol=1e-12)


def test_restoration_coupling_cannot_move_the_depth_metric():
    """The embargo makes this structural: Lambda is measured before any repair."""
    engine = build_case(n_supply=12, n_service=12)
    runs, _ = run_factorial(engine, base_scenario(), replications=6)
    effects = paired_effects(runs, metrics=("Lambda",))
    np.testing.assert_allclose(effects["Lambda__delta_rec"], 0.0, atol=1e-12)
