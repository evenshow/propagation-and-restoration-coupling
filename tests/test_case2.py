"""Tests for case 2: EPANET Net3 water network + Sioux Falls road network.

Case 2 is the first case whose damageable components (pipes) are disjoint from
its service nodes (junctions), so it is where the framework's component /
service-node separation is actually exercised — and where two engine bugs that
case 1 could not reveal were found.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.power_transport.geography import bounding_box
from coupling0729.cases.water_transport import base_scenario, build_case, case_weights
from coupling0729.core.scenario import ARM_NAMES


def _short(scenario, t_end: float = 460.0):
    return replace(scenario, t_end=t_end, T_eval=t_end - 100.0)


@pytest.fixture(scope="module")
def engine():
    return build_case()


# --- the separation that case 1 could not test ---------------------------
def test_components_and_service_nodes_are_disjoint_sets(engine):
    water = engine.layers["W"]
    components = set(water.components())
    services = set(water.service_nodes())
    junctions = {s for s in services if s.startswith("junction_")}
    assert components and junctions
    assert not (components & junctions), "pipes and junctions must be distinct"
    assert components <= services, "pipes stay in the performance vector as the interface"


def test_only_damageable_components_are_sampled(engine):
    """Regression: the engine used to hand every position to the hazard sampler,
    including service nodes, which have no fragility class."""
    positions, classes = engine._component_inventory()
    assert set(classes) < set(positions), "service nodes should have positions, not classes"
    assert all(c.startswith(("W:pipe_", "T:")) for c in classes)
    # Every junction has a position (coupling needs it) but no class.
    junctions = [p for p in positions if p.startswith("W:junction_")]
    assert junctions and not any(j in classes for j in junctions)


def test_hazard_sampler_rejects_unclassified_positions(engine):
    from coupling0729.core.hazard import sample_damage

    positions, classes = engine._component_inventory()
    with pytest.raises(KeyError, match="without a fragility class"):
        sample_damage(
            positions=positions,          # includes service nodes
            component_class=classes,
            fragility=engine.fragility,
            field_model=engine.hazard_field,
            window=base_scenario().window,
            seed_hazard=0,
        )


# --- geography ------------------------------------------------------------
def test_water_network_is_placed_inside_the_road_footprint(engine):
    water_box = bounding_box(engine.layers["W"].positions())
    road_box = bounding_box(engine.layers["T"].positions())
    assert water_box[0] >= road_box[0] - 1e-6
    assert water_box[1] <= road_box[1] + 1e-6
    assert water_box[2] >= road_box[2] - 1e-6
    assert water_box[3] <= road_box[3] + 1e-6


# --- hydraulics -----------------------------------------------------------
def test_closing_pipes_reduces_delivered_demand(engine):
    water = engine.layers["W"]
    water.reset(0)
    water.step(0.0, 10.0)
    before = float(np.mean([v for k, v in water.operational().items()
                            if k.startswith("junction_")]))

    for pipe in list(water.components())[:12]:
        water.apply_damage(pipe, capacity_factor=0.0, work=1.0)
    water.step(10.0, 10.0)
    after = float(np.mean([v for k, v in water.operational().items()
                           if k.startswith("junction_")]))
    assert after < before, "closing mains must cut delivered demand"


def test_hydraulics_are_cached_on_the_closed_pipe_set(engine):
    """Re-solving WNTR every step would dominate run time."""
    water = engine.layers["W"]
    water.reset(0)
    water.step(0.0, 10.0)
    key = water._cache_key
    water.step(10.0, 10.0)
    assert water._cache_key is key or water._cache_key == key

    pipe = list(water.components())[0]
    water.apply_damage(pipe, capacity_factor=0.0, work=1.0)
    water.step(20.0, 10.0)
    assert water._cache_key != key, "closing a pipe must invalidate the cache"


def test_closing_every_main_leaves_no_supply(engine):
    water = engine.layers["W"]
    water.reset(0)
    for pipe in water.components():
        water.apply_damage(pipe, capacity_factor=0.0, work=1.0)
    water.step(10.0, 10.0)
    service = [v for k, v in water.operational().items() if k.startswith("junction_")]
    assert service and max(service) == pytest.approx(0.0)


def test_hydraulic_infeasibility_degrades_instead_of_crashing(engine):
    """Regression: for some closure patterns WNTR returns an *empty* result set
    rather than raising, which used to surface as an opaque ``IndexError`` deep
    in pandas and killed whole simulation blocks.

    The fallback is tested directly rather than by hunting for a triggering
    configuration: total closure, the obvious candidate, is in fact solvable
    (no flow is a valid steady state), so the trigger is some intermediate
    pattern that is not worth pinning a test to.
    """
    water = engine.layers["W"]
    water.reset(0)
    water.step(0.0, 10.0)
    assert water.infeasible_solves == 0

    original = water._simulate
    water._simulate = lambda closed: None      # force an unsolvable step
    try:
        water._cache_key = None                # invalidate so _refresh re-solves
        water.step(10.0, 10.0)
    finally:
        water._simulate = original

    service = [v for k, v in water.operational().items() if k.startswith("junction_")]
    assert service, "junctions must still be reported"
    assert max(service) == pytest.approx(0.0), "infeasible means no supply"
    assert water.infeasible_solves == 1, "the fallback must be counted, not silent"


def test_feasible_hydraulics_do_not_trip_the_fallback(engine):
    water = engine.layers["W"]
    water.reset(0)
    water.step(0.0, 10.0)
    assert water.infeasible_solves == 0


def test_repair_reopens_a_closed_pipe(engine):
    water = engine.layers["W"]
    water.reset(0)
    pipe = list(water.components())[3]
    water.apply_damage(pipe, capacity_factor=0.0, work=1.0)
    assert pipe in water._closed_pipes() or water._bare(pipe) in water._closed_pipes()
    water.repair(pipe, 1.0)
    assert water.infrastructure()[pipe] == pytest.approx(1.0, abs=1e-6)
    assert not water.repair_backlog()


# --- weights --------------------------------------------------------------
def test_pipes_carry_no_system_weight(engine):
    weights = case_weights(engine)
    assert all(v == 0.0 for k, v in weights.items() if k.startswith("W:pipe_"))
    assert sum(v for k, v in weights.items() if k.startswith("W:")) == pytest.approx(0.5)
    assert sum(v for k, v in weights.items() if k.startswith("T:")) == pytest.approx(0.5)


# --- identification -------------------------------------------------------
def test_arms_see_identical_flood_damage(engine):
    scenario = _short(base_scenario()).for_replication(1)
    summaries = [engine.run(scenario.for_arm(a)).damage_summary for a in ARM_NAMES]
    assert all(s == summaries[0] for s in summaries)


def test_hazard_damages_both_layers(engine):
    result = engine.run(_short(base_scenario()).for_replication(0).for_arm("00"))
    infra = result.infrastructure.iloc[-1]
    assert any(v < 0.999 for k, v in infra.items() if k.startswith("W:"))
    assert any(v < 0.999 for k, v in infra.items() if k.startswith("T:"))


def test_restoration_coupling_cannot_move_the_depth_metric(engine):
    from coupling0729.metrics import result_metrics

    scenario = _short(base_scenario())
    weights = case_weights(engine)
    lam = {
        arm: result_metrics(
            engine.run(scenario.for_replication(2).for_arm(arm)),
            "operational", weights=weights,
        )["Lambda"]
        for arm in ("00", "01")
    }
    assert lam["00"] == pytest.approx(lam["01"], abs=1e-12)
