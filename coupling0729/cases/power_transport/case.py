"""Case 1: IEEE33 distribution feeder coupled with the Sioux Falls road network.

Hazard: **pluvial flooding**, not an earthquake. The choice is deliberate. The
framework accrues damage monotonically across a hazard window of non-zero width,
and rising floodwater is the hazard for which that is literally true -- a
component fails when the water reaches its threshold and does not un-fail.
Ground shaking lasts seconds, which would collapse Phase I to a cliff and make
the window structure vacuous. Flooding also matches the power-transport
rainstorm literature directly.

Interdependence, deliberately not symmetric:

``D_prop``  ``E:bus -> T:node``   power loss removes signal control, adding
                                  intersection delay and cutting effective road
                                  capacity;
``D_rec``   ``T:node -> E:bus``   repair crews must reach a flooded substation
                                  over the road network, so restoration is
                                  gated and derated by road service.

The two operate on different node sets in different phases and neither is the
reverse of the other.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ...core.coupling import PROPAGATION, RESTORATION
from ...core.damage import DamageMapping
from ...core.engine import CouplingSpec, PhaseAwareEngine
from ...core.hazard import LognormalFragility, RadialHazardField
from ...core.repair import ResourceConstrainedRepair
from ...core.scenario import CouplingDesign, HazardWindow, Scenario
from .geography import bounding_box, embed_feeder
from .layers import PowerLayer, TrafficLayer

# Fragility in metres of inundation depth. Illustrative values chosen so that
# the two classes have different thresholds -- roads become impassable at
# shallower depths than switchgear fails -- rather than calibrated to a specific
# asset inventory. A real study would substitute measured depth-damage curves;
# the framework treats fragility as a case plugin precisely so that this can be
# swapped without touching the engine.
FLOOD_FRAGILITY: Mapping[str, LognormalFragility] = {
    # Substation switchgear sits above grade, so it tolerates shallow water.
    "substation": LognormalFragility(
        medians=(0.30, 0.50, 0.80, 1.20), betas=(0.50, 0.50, 0.50, 0.50)
    ),
    # Road nodes lose passability earlier than switchgear fails -- standing
    # water slows traffic well before it destroys the carriageway -- but not
    # so much earlier that the road layer is destroyed at every severity for
    # which the feeder is damaged at all, which would leave no headroom in
    # which to observe propagation coupling.
    "road_node": LognormalFragility(
        medians=(0.25, 0.45, 0.70, 1.10), betas=(0.50, 0.50, 0.50, 0.50)
    ),
}


def build_case(
    flood_peak: float = 0.60,
    attenuation_km: float = 4.0,
    correlation_km: float = 2.0,
    sigma: float = 0.40,
    epicentre_km: tuple[float, float] = (1.0, -2.0),
    access_threshold: float = 0.4,
    gate_indicator: str = "operational",
    tau: float = 15.0,
    min_capacity_floor: float = 0.20,
) -> PhaseAwareEngine:
    """Assemble the IEEE33 + Sioux Falls engine on a shared geography.

    ``flood_peak`` is calibrated, not arbitrary. Below about 0.5 m the feeder
    is barely damaged, so restoration coupling -- which gates *power* repair --
    has nothing to act on; above about 0.8 m the radial feeder collapses and
    every indicator saturates. The default sits in the band where both layers
    are damaged and both retain headroom.
    """
    # A flooded intersection still passes some traffic; a floor of zero makes
    # the road network fragment and drives service to zero at any severity.
    traffic = TrafficLayer(tau=tau, min_capacity_floor=float(min_capacity_floor))
    road_positions = traffic.positions()

    power_stub = PowerLayer(positions={}, fragility_class="substation")
    feeder_edges = [
        (str(row.from_bus), str(row.to_bus))
        for row in power_stub.sim.net.line.itertuples()
    ]
    bus_positions = embed_feeder(
        edges=feeder_edges,
        nodes=power_stub.components(),
        target_box=bounding_box(road_positions),
    )
    power = PowerLayer(
        positions=bus_positions,
        net=power_stub.sim.net,
        fragility_class="substation",
    )

    return PhaseAwareEngine(
        layers={"E": power, "T": traffic},
        hazard_field=RadialHazardField(
            peak=float(flood_peak),
            attenuation=float(attenuation_km),
            sigma=float(sigma),
            corr_length=float(correlation_km),
            epicentre=tuple(epicentre_km),
        ),
        fragility=FLOOD_FRAGILITY,
        propagation_spec=CouplingSpec(
            phase=PROPAGATION, source_layer="E", target_layer="T", assignment="nearest"
        ),
        restoration_spec=CouplingSpec(
            phase=RESTORATION, source_layer="T", target_layer="E", assignment="nearest"
        ),
        damage_mapping=DamageMapping(),
        repair_model=ResourceConstrainedRepair(
            access_threshold=float(access_threshold), gate_indicator=str(gate_indicator)
        ),
        execution_order=["E", "T"],
        hazard_shape="ramp",
        warmup_steps=3,
    )


def case_weights(engine: PhaseAwareEngine, power_share: float = 0.5) -> dict[str, float]:
    """System-curve weights: load-weighted within a layer, balanced across layers.

    Two corrections to a plain node average, both standard rather than
    convenient. Following Panteli et al., the operational indicator of a power
    system is *connected load*, not the count of energised buses -- so a bus
    weight is its demand, and de-energising a zero-load junction costs nothing.
    And because the feeder has 33 buses against 24 road nodes, an unweighted mean
    would let the power layer dominate a metric that is supposed to describe both;
    each layer is therefore normalised to a fixed share.
    """
    power, traffic = engine.layers["E"], engine.layers["T"]

    load = power.sim.net.load.groupby("bus")["p_mw"].sum()
    power_raw = {}
    for bus in power.components():
        value = float(load.get(int(bus), 0.0))
        # A small floor keeps zero-load junctions from vanishing entirely: they
        # still carry power onwards, so they are not worthless.
        power_raw[f"E:{bus}"] = max(value, 0.02)

    net = traffic.sim.net
    od = np.asarray(net.od, dtype=float)
    activity = od.sum(axis=0) + od.sum(axis=1)
    traffic_raw = {
        f"T:{node}": max(float(activity[net.pos(node)]), 1e-6)
        for node in traffic.components()
    }

    weights = {}
    for raw, share in ((power_raw, power_share), (traffic_raw, 1.0 - power_share)):
        total = sum(raw.values())
        weights.update({k: share * v / total for k, v in raw.items()})
    return weights


def base_scenario(
    q_prop: float = 0.5,
    kappa_prop: float = 0.8,
    q_rec: float = 0.5,
    kappa_rec: float = 0.8,
    resources_per_step: float = 0.010,
    dt: float = 10.0,
    T_eval: float = 4000.0,
) -> Scenario:
    """Time is in minutes: a 4-hour rainfall event, 1-hour crew mobilisation.

    ``T_eval`` is set from the most strongly coupled arm rather than the
    uncoupled one, because censoring rises with coupling strength and a horizon
    tuned on arm 00 would censor arm 11 preferentially -- biasing the estimated
    effect towards zero.
    """
    return Scenario(
        window=HazardWindow(t_oe=60.0, t_ee=300.0, t_mob=60.0),
        design=CouplingDesign(
            q_prop=q_prop, kappa_prop=kappa_prop, q_rec=q_rec, kappa_rec=kappa_rec
        ),
        dt=float(dt),
        t_end=float(T_eval + 400.0),
        resources_per_step=float(resources_per_step),
        T_eval=float(T_eval),
        eta=0.01,
        eta_steps=2,
        tag="ieee33_siouxfalls",
    )
