"""Phase-specific coupling operators.

Two directed, weighted operators, deliberately **not** inverses of each other:

``D_prop``
    source-layer node -> target-layer node. Acts during the disruption phase on
    the target's *service*.
``D_rec``
    support-layer node -> damaged component of the target layer. Acts during
    the restoration phase on the target's *repair*, through two channels: it
    can postpone the start of repair, and it can derate the repair rate.

Both phases share one semantics for the edge weight:

    **kappa is the fraction of the source's degradation that is transmitted.**

An edge ``(i -> j, kappa)`` with source performance ``p_i`` transmits a
degradation of ``kappa * (1 - p_i)``. Hence ``kappa = 0`` is indistinguishable
from a missing edge, which is what makes the ablation arms corner points of the
same continuous design space rather than a separate mechanism.

A coupling operator is summarised by two orthogonal scalars: breadth ``q`` (the
share of target-layer nodes with at least one incoming edge) and intensity
``kappa_bar`` (the mean edge weight). Their product ``psi = q * kappa_bar`` is
the candidate sufficient statistic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

PROPAGATION = "propagation"
RESTORATION = "restoration"


@dataclass(frozen=True)
class CouplingEdge:
    source: str
    target: str
    kappa: float

    def transmitted(self, source_performance: float) -> float:
        """Degradation transmitted along this edge."""
        deficit = 1.0 - float(source_performance)
        if deficit <= 0.0:
            return 0.0
        return float(self.kappa) * deficit


@dataclass(frozen=True)
class CouplingOperator:
    """An immutable set of weighted dependency edges for one phase."""

    phase: str
    edges: tuple[CouplingEdge, ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in (PROPAGATION, RESTORATION):
            raise ValueError(f"unknown phase {self.phase!r}")
        for edge in self.edges:
            if not 0.0 <= float(edge.kappa) <= 1.0:
                raise ValueError(f"kappa must lie in [0, 1], got {edge.kappa}")

    # -- structure -------------------------------------------------------
    def incoming(self, target: str) -> tuple[CouplingEdge, ...]:
        return tuple(e for e in self.edges if e.target == target)

    def targets(self) -> set[str]:
        return {e.target for e in self.edges}

    def sources(self) -> set[str]:
        return {e.source for e in self.edges}

    def is_empty(self) -> bool:
        return len(self.edges) == 0 or all(e.kappa <= 0.0 for e in self.edges)

    def disabled(self) -> "CouplingOperator":
        return CouplingOperator(phase=self.phase, edges=())

    # -- transmission ----------------------------------------------------
    def support_level(self, target: str, performance: Mapping[str, float]) -> float:
        """Effective support ``1 - max_i kappa_i (1 - p_i)`` for one target.

        The maximum (rather than a sum or a noisy-OR) keeps the quantity in
        ``[0, 1]`` and makes a single fully-transmitting failed source
        sufficient to zero out support, which is the behaviour intended for a
        physical dependency.
        """
        worst = 0.0
        for edge in self.edges:
            if edge.target != target:
                continue
            worst = max(worst, edge.transmitted(float(performance.get(edge.source, 1.0))))
        return float(np.clip(1.0 - worst, 0.0, 1.0))

    def support_levels(
        self, targets: Iterable[str], performance: Mapping[str, float]
    ) -> dict[str, float]:
        return {t: self.support_level(t, performance) for t in targets}

    # -- descriptors -----------------------------------------------------
    def q(self, target_universe: Sequence[str]) -> float:
        universe = list(target_universe)
        if not universe:
            return 0.0
        covered = self.targets() & set(universe)
        return float(len(covered) / len(universe))

    def kappa_bar(self) -> float:
        if not self.edges:
            return 0.0
        return float(np.mean([e.kappa for e in self.edges]))

    def psi(self, target_universe: Sequence[str]) -> float:
        return float(self.q(target_universe) * self.kappa_bar())

    def summary(self, target_universe: Sequence[str]) -> dict:
        return {
            "phase": self.phase,
            "n_edges": len(self.edges),
            "n_targets_covered": len(self.targets() & set(target_universe)),
            "n_target_universe": len(list(target_universe)),
            "q": self.q(target_universe),
            "kappa_bar": self.kappa_bar(),
            "psi": self.psi(target_universe),
        }

    def signature(self) -> dict:
        return {
            "phase": self.phase,
            "edges": sorted(
                (e.source, e.target, round(float(e.kappa), 6)) for e in self.edges
            ),
        }


# ---------------------------------------------------------------------------
# Design-variable driven construction
# ---------------------------------------------------------------------------
def build_operator(
    phase: str,
    sources: Sequence[str],
    targets: Sequence[str],
    q: float,
    kappa: float,
    seed_design: int,
    placement: str = "random",
    priority: Mapping[str, float] | None = None,
    positions: Mapping[str, tuple[float, float]] | None = None,
    assignment: str = "nearest",
) -> CouplingOperator:
    """Construct an operator realising a target breadth ``q`` and intensity ``kappa``.

    ``placement`` selects *which* targets become dependent:

    ``random``   uniform sample (the neutral default);
    ``top``      the highest-``priority`` targets first (e.g. busiest nodes);
    ``spread``   evenly spaced through the priority ordering, giving the same
                 count with deliberately dispersed placement.

    Placement is drawn from ``seed_design`` only, so a design point denotes the
    same physical layout for every Monte Carlo replication.
    """
    targets = list(targets)
    sources = list(sources)
    if not targets or not sources:
        return CouplingOperator(phase=phase, edges=())
    if kappa <= 0.0 or q <= 0.0:
        return CouplingOperator(phase=phase, edges=())

    n_dependent = int(round(float(q) * len(targets)))
    n_dependent = int(np.clip(n_dependent, 0, len(targets)))
    if n_dependent == 0:
        return CouplingOperator(phase=phase, edges=())

    rng = np.random.default_rng(int(seed_design))
    chosen = _select_targets(targets, n_dependent, placement, priority, rng)
    pick = _source_picker(sources, positions, assignment, rng)

    edges = tuple(
        CouplingEdge(source=pick(target), target=target, kappa=float(kappa))
        for target in chosen
    )
    return CouplingOperator(phase=phase, edges=edges)


def _select_targets(
    targets: Sequence[str],
    count: int,
    placement: str,
    priority: Mapping[str, float] | None,
    rng: np.random.Generator,
) -> list[str]:
    if placement == "random":
        idx = rng.choice(len(targets), size=count, replace=False)
        return [targets[i] for i in sorted(idx)]

    scores = priority or {}
    ordered = sorted(targets, key=lambda t: (-float(scores.get(t, 0.0)), str(t)))
    if placement == "top":
        return ordered[:count]
    if placement == "spread":
        idx = np.linspace(0, len(ordered) - 1, num=count)
        return [ordered[int(round(i))] for i in idx]
    raise ValueError(f"unknown placement strategy {placement!r}")


def _source_picker(
    sources: Sequence[str],
    positions: Mapping[str, tuple[float, float]] | None,
    assignment: str,
    rng: np.random.Generator,
):
    if assignment == "random" or not positions:
        order = rng.permutation(len(sources))
        counter = {"i": 0}

        def pick_random(_target: str) -> str:
            source = sources[order[counter["i"] % len(sources)]]
            counter["i"] += 1
            return source

        return pick_random

    if assignment != "nearest":
        raise ValueError(f"unknown assignment strategy {assignment!r}")

    known = [s for s in sources if s in positions]
    if not known:
        return _source_picker(sources, None, "random", rng)
    coords = np.asarray([positions[s] for s in known], dtype=float)

    def pick_nearest(target: str) -> str:
        if target not in positions:
            return known[int(rng.integers(len(known)))]
        d = np.linalg.norm(coords - np.asarray(positions[target], dtype=float), axis=1)
        return known[int(np.argmin(d))]

    return pick_nearest
