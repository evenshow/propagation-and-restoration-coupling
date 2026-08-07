"""Resource-constrained repair with two restoration-coupling channels.

Restoration coupling acts on repair, not on service, and it acts through two
distinct channels that the literature usually conflates:

``start gating``
    a damaged component whose support level has collapsed cannot be worked on
    at all this step -- crews cannot reach it -- so repair is *postponed*;
``rate derating``
    a component that is reachable but poorly supported is repaired more slowly.

Both are driven by the same effective support ``s = 1 - max_i kappa_i (1 - p_i)``
produced by the restoration operator, so a single intensity parameter
``kappa_rec`` controls both, and ``kappa_rec = 0`` switches both off exactly.

Gated components do **not** consume crew capacity: crews redeploy to reachable
work. This matters -- charging gating against the resource budget would
double-count the effect and would make restoration coupling look like a simple
resource shortage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .coupling import CouplingOperator
from .naming import node_col

DEFAULT_ACCESS_THRESHOLD = 0.4


@dataclass(frozen=True)
class ResourceConstrainedRepair:
    """Greedy allocation of a fixed repair budget across damaged components.

    ``gate_indicator`` selects which state of the supporting layer the gate
    reads, and it is a substantive modelling choice rather than a detail:

    ``operational``
        support is read from delivered service. Realistic where access depends
        on the supporting network actually working, but it admits **lock-in**:
        if the supporting layer's service itself depends on the layer being
        repaired, neither can recover. Lock-in is measured and reported rather
        than assumed away.
    ``infrastructure``
        support is read from physical integrity. A damaged-but-passable
        supporting network still permits repair, so lock-in cannot occur.
    """

    access_threshold: float = DEFAULT_ACCESS_THRESHOLD
    strategy: str = "largest_backlog"
    gate_indicator: str = "operational"

    def signature(self) -> dict:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "access_threshold": float(self.access_threshold),
            "strategy": self.strategy,
            "gate_indicator": self.gate_indicator,
        }

    def step(
        self,
        t: float,
        dt: float,
        layers: Mapping[str, object],
        operator: CouplingOperator,
        performance: Mapping[str, float],
        resources: float,
        infrastructure: Mapping[str, float] | None = None,
    ) -> list[dict]:
        budget = float(resources) * float(dt)
        if budget <= 0.0:
            return []

        if self.gate_indicator == "operational":
            state = performance
        elif self.gate_indicator == "infrastructure":
            state = infrastructure if infrastructure is not None else performance
        else:
            raise ValueError(f"unknown gate_indicator {self.gate_indicator!r}")

        queue = []
        for layer_name, layer in layers.items():
            for component, backlog in layer.repair_backlog().items():
                if float(backlog) <= 1e-12:
                    continue
                col = node_col(layer_name, component)
                support = (
                    operator.support_level(col, state) if not operator.is_empty() else 1.0
                )
                queue.append((layer_name, layer, component, float(backlog), float(support), col))

        if self.strategy == "largest_backlog":
            queue.sort(key=lambda row: -row[3])
        elif self.strategy == "smallest_backlog":
            queue.sort(key=lambda row: row[3])
        else:
            raise ValueError(f"unknown repair strategy {self.strategy!r}")

        events: list[dict] = []
        for layer_name, layer, component, backlog, support, col in queue:
            if budget <= 1e-12:
                break
            if support < self.access_threshold:
                events.append(
                    {"t": float(t), "type": "repair_postponed", "target": col,
                     "support": round(support, 4)}
                )
                continue
            allocation = min(budget, backlog / max(support, 1e-9))
            consumed = layer.repair(component, allocation * support)
            budget -= allocation
            if consumed > 0.0:
                events.append(
                    {
                        "t": float(t),
                        "type": "repair_derated" if support < 1.0 else "repair",
                        "target": col,
                        "work": round(float(consumed), 6),
                        "support": round(support, 4),
                    }
                )
        return events
