"""A minimal, fully transparent two-layer case for validating the framework.

Layer ``A`` is a supply layer: its service is purely physical, so its
operational and infrastructure indicators coincide. Layer ``B`` is a service
layer whose delivered service depends both on its own physical capacity and on
support received from ``A``. ``B`` holds a buffer that masks a supply deficit
until it drains, which gives the propagation channel a realistic time constant
instead of an instantaneous algebraic link.

Restoration coupling runs the other way and is *not* the reverse of the
propagation edge: repair of ``A``'s damaged components is gated by ``B``'s
service (crews and materials have to move through ``B`` to reach the site).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ...core.coupling import PROPAGATION, RESTORATION
from ...core.engine import CouplingSpec, PhaseAwareEngine
from ...core.hazard import LognormalFragility, RadialHazardField
from ...core.repair import ResourceConstrainedRepair
from ...core.scenario import CouplingDesign, HazardWindow, Scenario


@dataclass
class GenericLayer:
    """One layer: components take damage, service nodes report performance.

    Components and service nodes are one-to-one here, which keeps both
    indicators directly readable:

    ``infrastructure`` = residual physical capacity (only repair restores it);
    ``operational``    = service actually delivered (cross-layer support and the
                         buffer can degrade or restore it without any physical
                         change).
    """

    layer: str
    n_nodes: int
    grid_origin: tuple[float, float] = (0.0, 0.0)
    grid_step: float = 1.5
    grid_width: int = 5
    fragility_class: str = "generic"
    buffer_capacity: float = 0.0
    buffer_recharge: float = 0.05

    capacity: dict = field(default_factory=dict, init=False)
    damaged_capacity: dict = field(default_factory=dict, init=False)
    backlog: dict = field(default_factory=dict, init=False)
    total_work: dict = field(default_factory=dict, init=False)
    support: dict = field(default_factory=dict, init=False)
    buffer: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.reset(0)

    # -- identity --------------------------------------------------------
    def node_ids(self) -> list[str]:
        return [f"{i}" for i in range(self.n_nodes)]

    def components(self) -> Sequence[str]:
        return self.node_ids()

    def service_nodes(self) -> Sequence[str]:
        return self.node_ids()

    def component_class(self, component: str) -> str:
        return self.fragility_class

    def positions(self) -> Mapping[str, tuple[float, float]]:
        out = {}
        for i, node in enumerate(self.node_ids()):
            row, col = divmod(i, self.grid_width)
            out[node] = (
                self.grid_origin[0] + col * self.grid_step,
                self.grid_origin[1] + row * self.grid_step,
            )
        return out

    def signature(self) -> Mapping:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "layer": self.layer,
            "n_nodes": self.n_nodes,
            "buffer_capacity": self.buffer_capacity,
            "buffer_recharge": self.buffer_recharge,
            "grid": [self.grid_origin, self.grid_step, self.grid_width],
        }

    # -- lifecycle -------------------------------------------------------
    def reset(self, seed: int) -> None:
        nodes = self.node_ids()
        self.capacity = {n: 1.0 for n in nodes}
        self.damaged_capacity = {n: 1.0 for n in nodes}
        self.backlog = {n: 0.0 for n in nodes}
        self.total_work = {n: 0.0 for n in nodes}
        self.support = {n: 1.0 for n in nodes}
        self.buffer = {n: float(self.buffer_capacity) for n in nodes}

    def apply_damage(self, component: str, capacity_factor: float, work: float) -> None:
        self.damaged_capacity[component] = float(capacity_factor)
        self.backlog[component] += float(work)
        self.total_work[component] += float(work)
        self._refresh_capacity(component)

    def set_cross_layer_support(self, support: Mapping[str, float]) -> None:
        for node, value in support.items():
            if node in self.support:
                self.support[node] = float(np.clip(value, 0.0, 1.0))

    def step(self, t: float, dt: float) -> None:
        if self.buffer_capacity <= 0.0:
            return
        for node in self.node_ids():
            deficit = 1.0 - self.support[node]
            if deficit > 1e-12:
                drawn = min(self.buffer[node], deficit * float(dt))
                self.buffer[node] -= drawn
            else:
                self.buffer[node] = min(
                    self.buffer_capacity, self.buffer[node] + self.buffer_recharge * float(dt)
                )

    # -- indicators ------------------------------------------------------
    def infrastructure(self) -> Mapping[str, float]:
        return dict(self.capacity)

    def operational(self) -> Mapping[str, float]:
        out = {}
        for node in self.node_ids():
            availability = self.support[node]
            if self.buffer_capacity > 0.0:
                availability = min(
                    1.0, availability + self.buffer[node] / self.buffer_capacity
                )
            out[node] = float(self.capacity[node] * availability)
        return out

    # -- repair ----------------------------------------------------------
    def repair_backlog(self) -> Mapping[str, float]:
        return {n: v for n, v in self.backlog.items() if v > 1e-12}

    def repair(self, component: str, work: float) -> float:
        available = self.backlog.get(component, 0.0)
        consumed = float(min(available, max(float(work), 0.0)))
        if consumed <= 0.0:
            return 0.0
        self.backlog[component] = available - consumed
        self._refresh_capacity(component)
        return consumed

    def _refresh_capacity(self, component: str) -> None:
        total = self.total_work[component]
        damaged = self.damaged_capacity[component]
        if total <= 1e-12:
            self.capacity[component] = float(damaged)
            return
        progress = 1.0 - self.backlog[component] / total
        self.capacity[component] = float(
            np.clip(damaged + (1.0 - damaged) * progress, 0.0, 1.0)
        )


# ---------------------------------------------------------------------------
def build_case(
    n_supply: int = 20,
    n_service: int = 20,
    buffer_capacity: float = 8.0,
    access_threshold: float = 0.4,
) -> PhaseAwareEngine:
    """Assemble the two-layer engine."""
    supply = GenericLayer(
        layer="A",
        n_nodes=n_supply,
        grid_origin=(0.0, 0.0),
        fragility_class="supply",
    )
    service = GenericLayer(
        layer="B",
        n_nodes=n_service,
        grid_origin=(0.4, 0.4),
        fragility_class="service",
        buffer_capacity=buffer_capacity,
    )

    fragility = {
        # Medians are in the same units as the hazard field's peak intensity.
        "supply": LognormalFragility(
            medians=(0.30, 0.50, 0.75, 1.05), betas=(0.45, 0.45, 0.45, 0.45)
        ),
        "service": LognormalFragility(
            medians=(0.45, 0.70, 1.00, 1.40), betas=(0.50, 0.50, 0.50, 0.50)
        ),
    }

    return PhaseAwareEngine(
        layers={"A": supply, "B": service},
        hazard_field=RadialHazardField(
            peak=1.0, attenuation=5.0, sigma=0.40, corr_length=2.5, epicentre=(3.0, 2.0)
        ),
        fragility=fragility,
        # Disruption propagates A -> B (service of B depends on supply from A).
        propagation_spec=CouplingSpec(
            phase=PROPAGATION, source_layer="A", target_layer="B", assignment="nearest"
        ),
        # Restoration of A's components is gated by B's service. Independent
        # edge set, not the reverse of the propagation edge.
        restoration_spec=CouplingSpec(
            phase=RESTORATION, source_layer="B", target_layer="A", assignment="nearest"
        ),
        repair_model=ResourceConstrainedRepair(access_threshold=access_threshold),
        execution_order=["A", "B"],
        hazard_shape="ramp",
        warmup_steps=5,
    )


def base_scenario(
    q_prop: float = 0.6,
    kappa_prop: float = 0.8,
    q_rec: float = 0.6,
    kappa_rec: float = 0.8,
    resources_per_step: float = 0.10,
) -> Scenario:
    """Default scenario: 24-unit hazard, 12-unit mobilisation, 900-unit horizon.

    ``T_eval`` is set from the *most strongly coupled* arm, not the uncoupled
    one: censoring rises with coupling strength, so a horizon tuned on arm 00
    censors arm 11 preferentially and would bias the estimated effect towards
    zero -- exactly the effect the study is trying to measure.
    """
    return Scenario(
        window=HazardWindow(t_oe=20.0, t_ee=44.0, t_mob=12.0),
        design=CouplingDesign(
            q_prop=q_prop, kappa_prop=kappa_prop, q_rec=q_rec, kappa_rec=kappa_rec
        ),
        dt=1.0,
        t_end=960.0,
        resources_per_step=resources_per_step,
        T_eval=900.0,
        eta=0.01,
        eta_steps=2,
        tag="toy2layer",
    )
