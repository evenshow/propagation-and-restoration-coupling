"""Damage state to physical consequence: residual capacity and repair work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .hazard import DamageState

# Residual capacity retained in each damage state, and the repair work needed
# to return to full function (in abstract resource-time units). The defaults
# follow the usual conservative engineering tables: slight damage degrades
# service but keeps the component usable, extensive and complete damage take it
# out of service entirely and differ only in how long they take to repair.
DEFAULT_CAPACITY = {
    DamageState.NONE: 1.0,
    DamageState.SLIGHT: 0.70,
    DamageState.MODERATE: 0.30,
    DamageState.EXTENSIVE: 0.0,
    DamageState.COMPLETE: 0.0,
}

DEFAULT_REPAIR_WORK = {
    DamageState.NONE: 0.0,
    DamageState.SLIGHT: 0.25,
    DamageState.MODERATE: 0.60,
    DamageState.EXTENSIVE: 1.00,
    DamageState.COMPLETE: 1.60,
}


@dataclass(frozen=True)
class DamageMapping:
    """Case-pluggable map from damage state to capacity and repair work."""

    capacity: Mapping[DamageState, float] = None
    repair_work: Mapping[DamageState, float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity", dict(self.capacity or DEFAULT_CAPACITY))
        object.__setattr__(
            self, "repair_work", dict(self.repair_work or DEFAULT_REPAIR_WORK)
        )
        for state in DamageState:
            if state not in self.capacity or state not in self.repair_work:
                raise ValueError(f"damage mapping is missing state {state.name}")

    def capacity_factor(self, state: DamageState) -> float:
        return float(self.capacity[DamageState(state)])

    def work_for(self, state: DamageState) -> float:
        return float(self.repair_work[DamageState(state)])

    def signature(self) -> dict:
        return {
            "capacity": {s.name: float(v) for s, v in sorted(self.capacity.items())},
            "repair_work": {s.name: float(v) for s, v in sorted(self.repair_work.items())},
        }
