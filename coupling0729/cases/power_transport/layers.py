"""Layer adapters wrapping the mature IEEE33 and Sioux Falls simulators.

The physics is reused, not reimplemented: ``PandapowerPowerSim`` carries AC power
flow with a thermal/voltage cascade, and ``TNTPTrafficSim`` carries user
equilibrium with BPR link costs. Both are wrapped rather than subclassed, so
this package owns the authoritative damage state and the legacy objects are
driven purely through their public surface.

Two adaptations matter.

**Damage is split.** The legacy simulators use a single ``damage`` value that
repair drains directly. The new protocol separates *residual capacity* (set once
when damage occurs) from *repair work* (drained over time), so that capacity
recovers as the backlog clears. The adapters keep that bookkeeping and write a
derived value into the legacy ``damage`` dict.

**Cross-layer support enters through capacity, not through damage.** Power
degradation at a road node represents loss of signal control and hence added
delay, following the standard functional dependency of traffic signals on
electricity. It reduces road capacity but is *not* physical damage, so it moves
the operational indicator while leaving the infrastructure indicator untouched.
That separation is what makes the two indicators independently informative.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _ensure_legacy_path() -> Path:
    """Put the legacy ``coupling_sim`` package on the import path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "coupling_sim"
        if candidate.exists():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            return candidate
    raise FileNotFoundError("cannot locate the legacy coupling_sim package")


LEGACY_ROOT = _ensure_legacy_path()

from coupling_sim.src.simulation.power_sim import (  # noqa: E402
    PandapowerPowerSim,
    load_pandapower_network,
)
from coupling_sim.src.simulation.traffic_sim import SiouxFallsTrafficSim  # noqa: E402

SIOUXFALLS_DATA = LEGACY_ROOT / "data" / "siouxfalls"


class _BacklogMixin:
    """Shared residual-capacity / repair-work bookkeeping."""

    def _init_backlog(self, components: Sequence[str]) -> None:
        self._damaged_capacity = {c: 1.0 for c in components}
        self._backlog = {c: 0.0 for c in components}
        self._total_work = {c: 0.0 for c in components}

    def apply_damage(self, component: str, capacity_factor: float, work: float) -> None:
        component = str(component)
        self._damaged_capacity[component] = float(capacity_factor)
        self._backlog[component] += float(work)
        self._total_work[component] += float(work)
        self._sync(component)

    def repair(self, component: str, work: float) -> float:
        component = str(component)
        available = self._backlog.get(component, 0.0)
        consumed = float(min(available, max(float(work), 0.0)))
        if consumed <= 0.0:
            return 0.0
        self._backlog[component] = available - consumed
        self._sync(component)
        return consumed

    def repair_backlog(self) -> Mapping[str, float]:
        return {c: v for c, v in self._backlog.items() if v > 1e-12}

    def physical_damage(self, component: str) -> float:
        """残余物理损伤: 1 - current capacity, recovering as the backlog drains."""
        total = self._total_work[component]
        damaged = self._damaged_capacity[component]
        if total <= 1e-12:
            return float(np.clip(1.0 - damaged, 0.0, 1.0))
        remaining = self._backlog[component] / total
        return float(np.clip((1.0 - damaged) * remaining, 0.0, 1.0))

    def _sync(self, component: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class PowerLayer(_BacklogMixin):
    """IEEE33 distribution feeder. Source layer for propagation."""

    layer = "E"

    def __init__(
        self,
        positions: Mapping[str, tuple[float, float]],
        net=None,
        fragility_class: str = "substation",
        close_tie_lines: bool = True,
        energise_threshold: float = 0.2,
    ) -> None:
        self.sim = PandapowerPowerSim(
            net=net if net is not None else load_pandapower_network("case33bw"),
            close_tie_lines=close_tie_lines,
        )
        self._positions = {str(k): (float(v[0]), float(v[1])) for k, v in positions.items()}
        self.fragility_class = str(fragility_class)
        self.energise_threshold = float(energise_threshold)
        self._init_backlog(self.components())

    # -- identity --------------------------------------------------------
    def components(self) -> Sequence[str]:
        return [str(b) for b in self.sim.node_ids()]

    def service_nodes(self) -> Sequence[str]:
        return self.components()

    def component_class(self, component: str) -> str:
        return self.fragility_class

    def positions(self) -> Mapping[str, tuple[float, float]]:
        return dict(self._positions)

    def signature(self) -> Mapping:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "power": dict(self.sim.signature()),
            "fragility_class": self.fragility_class,
            "positions": {k: [round(v[0], 4), round(v[1], 4)]
                          for k, v in sorted(self._positions.items())},
        }

    # -- lifecycle -------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.sim.reset(int(seed))
        self._init_backlog(self.components())

    def set_cross_layer_support(self, support: Mapping[str, float]) -> None:
        """No inbound propagation in this case: power is the source layer."""

    def step(self, t: float, dt: float) -> None:
        self.sim.step(float(t), float(dt))

    # -- indicators ------------------------------------------------------
    def operational(self) -> Mapping[str, float]:
        """Energisation from the solver, times this adapter's residual capacity."""
        energised = self.sim.node_functions()
        out = {}
        for component in self.components():
            on = 1.0 if float(energised.get(component, 0.0)) > 0.0 else 0.0
            out[component] = float(on * (1.0 - self.physical_damage(component)))
        return out

    def infrastructure(self) -> Mapping[str, float]:
        return {c: 1.0 - self.physical_damage(c) for c in self.components()}

    def _sync(self, component: str) -> None:
        """Write only a binary outage into the legacy solver.

        The legacy model derates *adjacent line capacity* by ``1 - damage``, on a
        base rating of only ~1.5x the pre-event current. Writing partial bus
        damage there therefore turns light flooding into a feeder-wide overload
        cascade: a single moderately damaged bus derates its lines below their
        pre-event flow and trips them. Physically a flooded switchgear does not
        reduce the thermal rating of the conductors leaving it.

        So partial damage is carried in this adapter's own capacity term (and
        appears in both indicators), while the solver is told only whether the
        bus is energisable. A bus stays out until repair restores it past
        ``energise_threshold`` -- a partially repaired substation is not
        re-energised. The default sits below the MODERATE residual capacity
        so that a moderately damaged substation runs derated rather than
        being disconnected; disconnecting it instead makes the radial feeder
        collapse over a very narrow severity band, leaving no usable range.
        """
        residual = 1.0 - self.physical_damage(component)
        self.sim.damage[component] = 0.0 if residual >= self.energise_threshold else 1.0


class TrafficLayer(_BacklogMixin):
    """Sioux Falls road network. Target of propagation, source for restoration."""

    layer = "T"

    def __init__(
        self,
        data_dir=None,
        fragility_class: str = "road_node",
        tau: float = 15.0,
        support_quantum: float = 0.02,
        min_capacity_floor: float = 0.05,
    ) -> None:
        self.sim = SiouxFallsTrafficSim(
            str(data_dir or SIOUXFALLS_DATA),
            stations={},  # propagation acts through the signal/capacity channel
            tau=float(tau),
            min_capacity_floor=float(min_capacity_floor),
        )
        self.fragility_class = str(fragility_class)
        self.support_quantum = float(support_quantum)
        self._support = {c: 1.0 for c in self.components()}
        self._init_backlog(self.components())

    # -- identity --------------------------------------------------------
    def components(self) -> Sequence[str]:
        return [str(n) for n in self.sim.node_ids()]

    def service_nodes(self) -> Sequence[str]:
        return self.components()

    def component_class(self, component: str) -> str:
        return self.fragility_class

    def positions(self) -> Mapping[str, tuple[float, float]]:
        from .geography import project_to_km

        return project_to_km({str(k): v for k, v in self.sim.net.coords.items()})

    def signature(self) -> Mapping:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "traffic": dict(self.sim.signature()),
            "fragility_class": self.fragility_class,
            "support_quantum": self.support_quantum,
        }

    # -- lifecycle -------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.sim.reset(int(seed))
        self._support = {c: 1.0 for c in self.components()}
        self._init_backlog(self.components())

    def set_cross_layer_support(self, support: Mapping[str, float]) -> None:
        """Store incoming support, quantised so the UE solve can still cache.

        Without quantisation a continuously drifting support vector would
        invalidate the legacy solver's signature cache at every step and force a
        fresh equilibrium solve, which dominates run time.
        """
        step = max(self.support_quantum, 1e-9)
        for node, value in support.items():
            key = str(node)
            if key in self._support:
                self._support[key] = float(round(float(value) / step) * step)

    def step(self, t: float, dt: float) -> None:
        # Physical damage and power-induced degradation combine into the single
        # availability figure the legacy solver understands. Our own state stays
        # authoritative, so this derived value can be overwritten freely.
        for component in self.components():
            physical = self.physical_damage(component)
            support = float(np.clip(self._support.get(component, 1.0), 0.0, 1.0))
            self.sim.damage[component] = float(
                np.clip(1.0 - (1.0 - physical) * support, 0.0, 1.0)
            )
        self.sim.step(float(t), float(dt), charging_capacity={})

    # -- indicators ------------------------------------------------------
    def operational(self) -> Mapping[str, float]:
        return {str(k): float(v) for k, v in self.sim.node_functions().items()}

    def infrastructure(self) -> Mapping[str, float]:
        # Physical integrity only: loss of signal power is not structural damage.
        return {c: 1.0 - self.physical_damage(c) for c in self.components()}

    def _sync(self, component: str) -> None:
        """Legacy damage is refreshed in :meth:`step`; nothing to do here."""
