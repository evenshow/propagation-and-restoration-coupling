"""Contracts a case must satisfy to run inside the generic engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class LayerSimulator(Protocol):
    """One infrastructure layer.

    A layer distinguishes *components* (physical assets that take damage and
    are repaired) from *service nodes* (where the operational indicator is
    read). They may coincide, but keeping them apart is what allows the two
    resilience indicators -- operational and infrastructure -- to be read off
    the same simulation.
    """

    layer: str

    def reset(self, seed: int) -> None:
        """Return to the undamaged pre-event state."""

    def components(self) -> Sequence[str]:
        """Identifiers of damageable components (without the layer prefix)."""

    def component_class(self, component: str) -> str:
        """Fragility class of a component."""

    def positions(self) -> Mapping[str, tuple[float, float]]:
        """Planar coordinates of components and service nodes."""

    def service_nodes(self) -> Sequence[str]:
        """Identifiers of nodes carrying the operational indicator."""

    def apply_damage(self, component: str, capacity_factor: float, work: float) -> None:
        """Record new damage: residual capacity and the repair work it adds.

        Capacity is expected to recover from ``capacity_factor`` back towards
        1.0 as the component's repair backlog drains, so the layer -- not the
        engine -- owns the interpolation between damaged and restored states.
        """

    def set_cross_layer_support(self, support: Mapping[str, float]) -> None:
        """Apply incoming propagation support levels, keyed by service node."""

    def step(self, t: float, dt: float) -> None:
        """Advance internal state by one step."""

    def operational(self) -> Mapping[str, float]:
        """Service level per service node."""

    def infrastructure(self) -> Mapping[str, float]:
        """Physical integrity per component, in ``[0, 1]``."""

    def repair(self, component: str, work: float) -> float:
        """Apply repair work; return the work actually consumed."""

    def repair_backlog(self) -> Mapping[str, float]:
        """Remaining repair work per component."""

    def signature(self) -> Mapping:
        """Hashable description used for run caching."""


@runtime_checkable
class HazardFieldModel(Protocol):
    def intensities(self, positions, rng) -> Mapping[str, float]:
        ...
