"""Water layer: EPANET Net3 through WNTR, pressure-dependent demand.

Unlike case 1, where a bus was both the damageable component and the service
node, here the two sets differ: **pipes** take damage and are repaired, while
**junctions** deliver service. That is the configuration the framework's
component / service-node separation was built for, and case 2 is where it is
actually exercised.

Pipes also appear in the performance vector, because they are the propagation
interface -- a burst main is what tears up the road, not a drop in pressure
somewhere downstream. Their "service" is conveyance integrity. The system curve
weights only demand-bearing junctions, so this does not inflate the water
layer's contribution (see ``case.case_weights``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..power_transport.layers import _BacklogMixin

# The network is shipped with this package rather than taken from whatever the
# installed wntr happens to bundle. Two reasons, both found the hard way:
# `WaterNetworkModel("Net3")` resolves a bundled name only from wntr 1.5, while
# wntr 1.0 treats the argument as a file path and does not ship Net3 at all; and
# more importantly, results must not depend on which wntr version is installed.
NET3_INP = Path(__file__).resolve().parents[3] / "data" / "net3" / "Net3.inp"


class WaterLayer(_BacklogMixin):
    """WNTR-backed Net3 water network."""

    layer = "W"

    def __init__(
        self,
        fragility_class: str = "water_pipe",
        close_threshold: float = 0.5,
        pressure_floor: float = 1e-9,
    ) -> None:
        import wntr

        self.wntr = wntr
        self.fragility_class = str(fragility_class)
        # A pipe is taken out of the hydraulic model once its residual capacity
        # falls below this; partial damage is carried in the adapter's own
        # capacity term, exactly as for the power layer.
        self.close_threshold = float(close_threshold)
        self.pressure_floor = float(pressure_floor)

        template = self._new_network()
        self.junction_ids = [str(j) for j in template.junction_name_list]
        self.pipe_ids = [str(p) for p in template.pipe_name_list]
        self._coordinates = self._collect_coordinates(template)

        self.infeasible_solves = 0
        baseline = self._simulate(frozenset())
        if baseline is None:
            raise RuntimeError("Net3 baseline hydraulics failed to solve")
        self._baseline_demand, self._baseline_pressure = baseline
        self._cache_key: frozenset | None = None
        self._junction_service: dict[str, float] = {}
        self._init_backlog(self.pipe_ids)
        self.reset(0)

    # -- network -----------------------------------------------------------
    def _new_network(self):
        if not NET3_INP.exists():
            raise FileNotFoundError(
                f"Net3 input file not found at {NET3_INP}. It is shipped with this "
                "package deliberately; falling back to wntr's bundled copy would "
                "make results depend on the installed wntr version."
            )
        wn = self.wntr.network.WaterNetworkModel(str(NET3_INP))
        wn.options.time.duration = 0
        wn.options.time.hydraulic_timestep = 3600
        wn.options.hydraulic.demand_model = "PDD"
        return wn

    def _collect_coordinates(self, network) -> dict[str, tuple[float, float]]:
        """Junction coordinates, plus pipe midpoints from their end nodes."""
        out: dict[str, tuple[float, float]] = {}
        for junction in self.junction_ids:
            x, y = network.get_node(junction).coordinates
            out[f"junction_{junction}"] = (float(x), float(y))

        node_xy: dict[str, tuple[float, float]] = {}
        for name in network.node_name_list:
            x, y = network.get_node(name).coordinates
            node_xy[str(name)] = (float(x), float(y))
        for pipe in self.pipe_ids:
            link = network.get_link(pipe)
            start = node_xy.get(str(link.start_node_name), (0.0, 0.0))
            end = node_xy.get(str(link.end_node_name), (0.0, 0.0))
            out[f"pipe_{pipe}"] = (0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]))
        return out

    def _simulate(
        self, closed: frozenset
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Solve hydraulics with ``closed`` pipes out of service.

        Returns ``None`` when the solver cannot find a steady state. With enough
        mains closed WNTR returns an empty result set rather than raising, so
        emptiness has to be checked explicitly -- left unchecked it surfaces
        much later as an opaque ``IndexError``.
        """
        from wntr.network import LinkStatus

        wn = self._new_network()
        for pipe in closed:
            if pipe in wn.pipe_name_list:
                wn.get_link(pipe).initial_status = LinkStatus.Closed
        if closed:
            # Load-bearing across wntr versions. On wntr 1.5 setting
            # `initial_status` alone is enough; on wntr 1.0 it is silently
            # ignored by the solver -- closing every one of Net3's 117 pipes
            # still leaves junction demand at 100% of baseline -- because the
            # status the model builds from is only refreshed here. Without this
            # call the whole damage mechanism is inert and the run completes
            # normally with meaningless numbers.
            wn.reset_initial_values()
        try:
            results = self.wntr.sim.WNTRSimulator(wn).run_sim()
            demand_frame = results.node["demand"]
            pressure_frame = results.node["pressure"]
        except Exception:
            return None
        if len(demand_frame) == 0 or len(pressure_frame) == 0:
            return None
        return (
            {str(k): float(v) for k, v in demand_frame.iloc[0].to_dict().items()},
            {str(k): float(v) for k, v in pressure_frame.iloc[0].to_dict().items()},
        )

    # -- identity ----------------------------------------------------------
    def components(self) -> Sequence[str]:
        return [f"pipe_{p}" for p in self.pipe_ids]

    def service_nodes(self) -> Sequence[str]:
        return [f"junction_{j}" for j in self.junction_ids] + list(self.components())

    def component_class(self, component: str) -> str:
        return self.fragility_class

    def positions(self) -> Mapping[str, tuple[float, float]]:
        return dict(self._coordinates)

    def signature(self) -> Mapping:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "network": "EPANET Net3",
            "n_junctions": len(self.junction_ids),
            "n_pipes": len(self.pipe_ids),
            "close_threshold": self.close_threshold,
        }

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: int) -> None:
        self._init_backlog(self.pipe_ids)
        self._cache_key = None
        self.infeasible_solves = 0
        self._refresh()

    def set_cross_layer_support(self, support: Mapping[str, float]) -> None:
        """No inbound propagation in this case: water is the source layer."""

    def step(self, t: float, dt: float) -> None:
        self._refresh()

    def _closed_pipes(self) -> frozenset:
        return frozenset(
            pipe for pipe in self.pipe_ids
            if 1.0 - self.physical_damage(pipe) < self.close_threshold
        )

    def _refresh(self) -> None:
        """Re-solve hydraulics only when the set of closed pipes changes.

        Closure is binary, so the signature changes far less often than the
        simulation steps; without this the WNTR solve would dominate run time.
        """
        key = self._closed_pipes()
        if key == self._cache_key:
            return
        self._cache_key = key
        solved = self._simulate(key)
        if solved is None:
            # Hydraulic infeasibility: with this many mains closed the solver
            # finds no steady state. Treated as total loss of supply, mirroring
            # the blackout fallback on the power side. Counted rather than
            # silently absorbed, because a case in which this fires often is
            # not measuring what it claims to.
            self.infeasible_solves += 1
            self._junction_service = {f"junction_{j}": 0.0 for j in self.junction_ids}
            return
        demand, pressure = solved
        service = {}
        for junction in self.junction_ids:
            base = float(self._baseline_demand.get(junction, 0.0))
            if base > 1e-9:
                value = demand.get(junction, 0.0) / base
            else:
                base_p = max(float(self._baseline_pressure.get(junction, 0.0)),
                             self.pressure_floor)
                value = pressure.get(junction, 0.0) / base_p
            service[f"junction_{junction}"] = float(np.clip(value, 0.0, 1.0))
        self._junction_service = service

    # -- indicators --------------------------------------------------------
    def operational(self) -> Mapping[str, float]:
        out = dict(self._junction_service)
        for pipe in self.pipe_ids:
            out[f"pipe_{pipe}"] = 1.0 - self.physical_damage(pipe)
        return out

    def infrastructure(self) -> Mapping[str, float]:
        return {f"pipe_{p}": 1.0 - self.physical_damage(p) for p in self.pipe_ids}

    # -- backlog bookkeeping keyed by bare pipe id -------------------------
    def apply_damage(self, component: str, capacity_factor: float, work: float) -> None:
        super().apply_damage(self._bare(component), capacity_factor, work)

    def repair(self, component: str, work: float) -> float:
        return super().repair(self._bare(component), work)

    def repair_backlog(self) -> Mapping[str, float]:
        return {f"pipe_{p}": v for p, v in super().repair_backlog().items()}

    def physical_damage(self, component: str) -> float:
        return super().physical_damage(self._bare(component))

    @staticmethod
    def _bare(component: str) -> str:
        text = str(component)
        return text[5:] if text.startswith("pipe_") else text

    def _sync(self, component: str) -> None:
        """Hydraulics are refreshed lazily in :meth:`step`; nothing to do here."""
