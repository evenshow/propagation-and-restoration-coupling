"""Case 2: EPANET Net3 water network coupled with the Sioux Falls road network.

The transfer test. Case 2 keeps the hazard, the design point, the metrics and
the experimental design identical to case 1 and changes only the *domain and its
physics* -- pressure-dependent hydraulics instead of AC power flow, buried pipes
instead of surface switchgear, and a component set (pipes) that is disjoint from
the service set (junctions). If the conclusions of case 1 reappear here, the
framework transfers; if they do not, the conditions under which they hold become
the finding.

Interdependence:

``D_prop``  ``W:pipe -> T:road_node``   a burst main tears up or floods the
                                        carriageway, cutting road capacity;
``D_rec``   ``T:road_node -> W:pipe``   repair crews must reach the break over
                                        the road network.

Note that this is structurally parallel to case 1, not a different topology:
both couple one layer pair in opposite directions across the two phases. The
contrast that matters is the physics, not the wiring.
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
from ..power_transport.geography import bounding_box, project_to_km
from ..power_transport.layers import TrafficLayer
from .layers import WaterLayer

# Fragility in metres of inundation depth, as in case 1. Buried mains are better
# protected than a road surface and than surface switchgear, so their thresholds
# sit highest; the failure mode represented is scour, joint separation and
# washout rather than direct submersion. Illustrative, not calibrated to an
# asset inventory.
FLOOD_FRAGILITY: Mapping[str, LognormalFragility] = {
    "water_pipe": LognormalFragility(
        medians=(0.35, 0.60, 0.90, 1.40), betas=(0.55, 0.55, 0.55, 0.55)
    ),
    "road_node": LognormalFragility(
        medians=(0.25, 0.45, 0.70, 1.10), betas=(0.50, 0.50, 0.50, 0.50)
    ),
}


def _fit_into(
    points: Mapping[str, tuple[float, float]],
    target_box: tuple[float, float, float, float],
    margin: float = 0.10,
) -> dict[str, tuple[float, float]]:
    """Affinely map one point set into another's bounding box.

    Net3 carries its own arbitrary coordinate system, so it has to be brought
    into the road network's frame before a spatially correlated hazard can hit
    both layers coherently. Unlike the IEEE33 feeder in case 1, Net3 does have
    coordinates, so this is a rescaling of a real layout rather than a
    synthesised one.
    """
    xs = np.array([p[0] for p in points.values()], dtype=float)
    ys = np.array([p[1] for p in points.values()], dtype=float)
    x_min, x_max, y_min, y_max = target_box
    pad_x, pad_y = margin * (x_max - x_min), margin * (y_max - y_min)

    def rescale(value, lo, hi, out_lo, out_hi):
        if hi - lo < 1e-12:
            return float(0.5 * (out_lo + out_hi))
        return float(out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo))

    return {
        name: (
            rescale(p[0], xs.min(), xs.max(), x_min + pad_x, x_max - pad_x),
            rescale(p[1], ys.min(), ys.max(), y_min + pad_y, y_max - pad_y),
        )
        for name, p in points.items()
    }


class _PlacedWaterLayer(WaterLayer):
    """Water layer whose coordinates have been mapped into the shared frame."""

    def __init__(self, placed: Mapping[str, tuple[float, float]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._placed = dict(placed)

    def positions(self) -> Mapping[str, tuple[float, float]]:
        return dict(self._placed)


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
    """Assemble the Net3 + Sioux Falls engine on a shared geography."""
    traffic = TrafficLayer(tau=tau, min_capacity_floor=float(min_capacity_floor))
    road_positions = traffic.positions()

    probe = WaterLayer()
    placed = _fit_into(probe.positions(), bounding_box(road_positions))
    water = _PlacedWaterLayer(placed)

    return PhaseAwareEngine(
        layers={"W": water, "T": traffic},
        hazard_field=RadialHazardField(
            peak=float(flood_peak),
            attenuation=float(attenuation_km),
            sigma=float(sigma),
            corr_length=float(correlation_km),
            epicentre=tuple(epicentre_km),
        ),
        fragility=FLOOD_FRAGILITY,
        propagation_spec=CouplingSpec(
            phase=PROPAGATION, source_layer="W", target_layer="T", assignment="nearest"
        ),
        restoration_spec=CouplingSpec(
            phase=RESTORATION, source_layer="T", target_layer="W", assignment="nearest"
        ),
        damage_mapping=DamageMapping(),
        repair_model=ResourceConstrainedRepair(
            access_threshold=float(access_threshold), gate_indicator=str(gate_indicator)
        ),
        execution_order=["W", "T"],
        hazard_shape="ramp",
        warmup_steps=3,
    )


def case_weights(engine: PhaseAwareEngine, water_share: float = 0.5) -> dict[str, float]:
    """Demand-weighted within a layer, balanced across layers.

    Pipes carry zero weight: they enter the performance vector only because they
    are the propagation interface, and counting a pipe's structural integrity as
    delivered service would double-count damage that already shows up as lost
    demand at the junctions it feeds.
    """
    water, traffic = engine.layers["W"], engine.layers["T"]

    water_raw = {
        f"W:junction_{j}": max(float(water._baseline_demand.get(j, 0.0)), 1e-9)
        for j in water.junction_ids
    }
    water_raw.update({f"W:pipe_{p}": 0.0 for p in water.pipe_ids})

    net = traffic.sim.net
    od = np.asarray(net.od, dtype=float)
    activity = od.sum(axis=0) + od.sum(axis=1)
    traffic_raw = {
        f"T:{node}": max(float(activity[net.pos(node)]), 1e-6)
        for node in traffic.components()
    }

    weights: dict[str, float] = {}
    for raw, share in ((water_raw, water_share), (traffic_raw, 1.0 - water_share)):
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
    T_eval: float = 7000.0,
) -> Scenario:
    """Same window and design point as case 1, but a longer evaluation horizon.

    Case 2 recovers considerably more slowly than case 1 (median ``T_rec`` in the
    uncoupled arm 2560 minutes against 1110), so case 1's horizon left 11% of the
    *uncoupled* runs censored. Censoring must be driven by coupling, not by the
    horizon, or the lock-in row measures the analyst's choice of ``T_eval``.
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
        tag="net3_siouxfalls",
    )
