"""Tests for case 1: IEEE33 feeder + Sioux Falls road network."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.power_transport import (
    base_scenario,
    build_case,
    case_weights,
)
from coupling0729.cases.power_transport.geography import (
    bounding_box,
    embed_feeder,
    project_to_km,
)
from coupling0729.core.hazard import DamageState
from coupling0729.core.scenario import ARM_NAMES


def _short(scenario, t_end: float = 420.0):
    """Truncate to just past the repair embargo, which is all most tests need."""
    return replace(scenario, t_end=t_end, T_eval=t_end - 80.0)


@pytest.fixture(scope="module")
def engine():
    return build_case()


# --- geography -------------------------------------------------------------
def test_projection_puts_the_city_at_a_plausible_scale():
    lonlat = {"a": (-96.79, 43.49), "b": (-96.69, 43.61)}
    km = project_to_km(lonlat)
    span_x = abs(km["b"][0] - km["a"][0])
    span_y = abs(km["b"][1] - km["a"][1])
    assert 5.0 < span_x < 12.0, "Sioux Falls is about 8 km across"
    assert 10.0 < span_y < 16.0


def test_both_layers_share_one_coordinate_frame(engine):
    """A spatially correlated hazard is meaningless if the layers are elsewhere."""
    power_box = bounding_box(engine.layers["E"].positions())
    road_box = bounding_box(engine.layers["T"].positions())
    # Feeder must sit inside the road network's footprint.
    assert power_box[0] >= road_box[0] - 1e-6
    assert power_box[1] <= road_box[1] + 1e-6
    assert power_box[2] >= road_box[2] - 1e-6
    assert power_box[3] <= road_box[3] + 1e-6


def test_feeder_embedding_is_deterministic():
    edges = [("0", "1"), ("1", "2"), ("2", "3"), ("1", "4")]
    nodes = ["0", "1", "2", "3", "4"]
    box = (0.0, 10.0, 0.0, 10.0)
    a = embed_feeder(edges, nodes, box)
    b = embed_feeder(edges, nodes, box)
    assert a == b


# --- the semantics-clash regression --------------------------------------
def test_partial_bus_damage_does_not_black_out_the_feeder(engine):
    """Regression: the legacy solver derates *adjacent line capacity* by
    ``1 - damage`` on a base rating of only ~1.5x pre-event current, so writing
    partial bus damage into it turned light flooding into a feeder-wide overload
    cascade. Partial damage must stay in the adapter's own capacity term."""
    power = engine.layers["E"]
    power.reset(0)
    power.step(0.0, 10.0)
    energised_before = int(sum(1 for v in power.operational().values() if v > 0.0))

    # Two buses at SLIGHT: capacity 0.70, well above the energisation threshold.
    for bus in list(power.components())[3:5]:
        power.apply_damage(bus, capacity_factor=0.70, work=0.25)
    power.step(10.0, 10.0)
    energised_after = int(sum(1 for v in power.operational().values() if v > 0.0))

    assert energised_after >= energised_before - 2, (
        f"light damage on 2 buses de-energised {energised_before - energised_after} "
        "buses: the cascade is being triggered by partial damage again"
    )


def test_severe_bus_damage_does_take_the_bus_out(engine):
    """The converse: a destroyed substation must actually leave service."""
    power = engine.layers["E"]
    power.reset(0)
    power.step(0.0, 10.0)
    target = list(power.components())[5]
    power.apply_damage(target, capacity_factor=0.0, work=1.0)
    power.step(10.0, 10.0)
    assert power.operational()[target] == pytest.approx(0.0)
    assert power.infrastructure()[target] == pytest.approx(0.0)


# --- indicator separation -------------------------------------------------
def test_cross_layer_support_moves_service_but_not_physical_integrity(engine):
    """Loss of signal power degrades road service without damaging the road."""
    traffic = engine.layers["T"]
    traffic.reset(0)
    for _ in range(3):
        traffic.step(0.0, 10.0)
    baseline = float(np.mean(list(traffic.operational().values())))
    infra_before = dict(traffic.infrastructure())

    traffic.set_cross_layer_support({c: 0.2 for c in traffic.components()})
    for step in range(12):
        traffic.step(10.0 * (step + 1), 10.0)

    degraded = float(np.mean(list(traffic.operational().values())))
    assert degraded < baseline - 0.02, "propagation support must degrade road service"
    assert traffic.infrastructure() == infra_before, (
        "power loss is not physical damage and must leave the infrastructure "
        "indicator untouched"
    )
    assert not traffic.repair_backlog(), "support must not create repair work"


def test_repair_restores_capacity_as_the_backlog_drains(engine):
    power = engine.layers["E"]
    power.reset(0)
    bus = list(power.components())[7]
    power.apply_damage(bus, capacity_factor=0.0, work=1.0)
    assert power.infrastructure()[bus] == pytest.approx(0.0)

    power.repair(bus, 0.5)
    assert power.infrastructure()[bus] == pytest.approx(0.5, abs=1e-6)
    power.repair(bus, 0.5)
    assert power.infrastructure()[bus] == pytest.approx(1.0, abs=1e-6)
    assert not power.repair_backlog()


# --- weights ---------------------------------------------------------------
def test_weights_are_layer_balanced_and_load_weighted(engine):
    weights = case_weights(engine)
    power = sum(v for k, v in weights.items() if k.startswith("E:"))
    traffic = sum(v for k, v in weights.items() if k.startswith("T:"))
    assert power == pytest.approx(0.5)
    assert traffic == pytest.approx(0.5)

    # A high-load bus must outweigh a zero-load junction.
    load = engine.layers["E"].sim.net.load.groupby("bus")["p_mw"].sum()
    heavy = str(int(load.idxmax()))
    zero = [b for b in engine.layers["E"].components() if int(b) not in load.index]
    if zero:
        assert weights[f"E:{heavy}"] > weights[f"E:{zero[0]}"]


# --- identification -------------------------------------------------------
def test_arms_see_identical_flood_damage(engine):
    scenario = _short(base_scenario()).for_replication(2)
    summaries = [engine.run(scenario.for_arm(arm)).damage_summary for arm in ARM_NAMES]
    assert all(s == summaries[0] for s in summaries)


def test_hazard_damages_both_layers(engine):
    """If the flood only reached one layer the case would be uninformative."""
    scenario = _short(base_scenario())
    result = engine.run(scenario.for_replication(0).for_arm("00"))
    infra = result.infrastructure.iloc[-1]
    power_damaged = any(v < 0.999 for k, v in infra.items() if k.startswith("E:"))
    road_damaged = any(v < 0.999 for k, v in infra.items() if k.startswith("T:"))
    assert power_damaged and road_damaged


def test_restoration_coupling_cannot_move_the_depth_metric(engine):
    """Structural, via the embargo: Lambda is read before any repair."""
    from coupling0729.metrics import result_metrics

    scenario = _short(base_scenario())
    weights = case_weights(engine)
    lam = {}
    for arm in ("00", "01"):
        result = engine.run(scenario.for_replication(1).for_arm(arm))
        lam[arm] = result_metrics(result, "operational", weights=weights)["Lambda"]
    assert lam["00"] == pytest.approx(lam["01"], abs=1e-12)
