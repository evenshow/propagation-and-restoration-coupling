"""Scenario description: hazard window, experimental arm, seeds, coupling design.

A ``Scenario`` is the complete, hashable description of one simulation run. The
fields are deliberately split so that the identification strategy is visible in
the data structure itself:

* ``window``  -- exogenous timing, identical across all arms;
* ``arm``     -- which coupling operators are switched on;
* ``seeds``   -- the three-level seed hierarchy (see :class:`SeedBundle`);
* ``design``  -- the manipulated coupling design variables (q, kappa).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

ARM_NAMES = ("00", "10", "01", "11")


@dataclass(frozen=True)
class HazardWindow:
    """Exogenous event timing shared by every experimental arm.

    ``t_oe``  hazard onset; ``t_ee`` hazard end; ``t_mob`` mobilisation delay
    before any repair may begin. No restoration takes place before
    ``t_repair_start``; this embargo is what makes the degraded plateau
    (Phase II) appear and keeps the depth metric free of repair contamination.
    """

    t_oe: float = 60.0
    t_ee: float = 84.0
    t_mob: float = 12.0

    @property
    def duration(self) -> float:
        return float(self.t_ee - self.t_oe)

    @property
    def t_repair_start(self) -> float:
        return float(self.t_ee + self.t_mob)

    def phase_of(self, t: float) -> str:
        if t < self.t_oe:
            return "pre"
        if t < self.t_ee:
            return "I"
        if t < self.t_repair_start:
            return "II"
        return "III"

    def validate(self) -> None:
        if not self.t_oe >= 0:
            raise ValueError("t_oe must be non-negative")
        if not self.t_ee > self.t_oe:
            raise ValueError("hazard window must have positive width (t_ee > t_oe)")
        if self.t_mob < 0:
            raise ValueError("t_mob must be non-negative")


@dataclass(frozen=True)
class Arm:
    """Which coupling operators are active in this run."""

    prop_on: bool
    rec_on: bool

    @property
    def name(self) -> str:
        return f"{int(self.prop_on)}{int(self.rec_on)}"

    @staticmethod
    def from_name(name: str) -> "Arm":
        text = str(name)
        if text not in ARM_NAMES:
            raise ValueError(f"unknown arm {name!r}; expected one of {ARM_NAMES}")
        return Arm(prop_on=text[0] == "1", rec_on=text[1] == "1")

    @staticmethod
    def all() -> tuple["Arm", ...]:
        return tuple(Arm.from_name(n) for n in ARM_NAMES)


@dataclass(frozen=True)
class SeedBundle:
    """Three-level seed hierarchy.

    ``hazard`` and ``sim`` are the common random numbers: they must be held
    identical across the four arms of one comparison so that paired differences
    isolate the coupling effect. ``design`` controls where coupling edges are
    placed and is held fixed across Monte Carlo replications so that a design
    point means the same thing for every hazard realisation.
    """

    hazard: int = 0
    sim: int = 0
    design: int = 0

    def with_replication(self, index: int) -> "SeedBundle":
        """Return the seed bundle for Monte Carlo replication ``index``."""
        return SeedBundle(hazard=int(index), sim=int(index), design=self.design)


@dataclass(frozen=True)
class CouplingDesign:
    """Manipulated coupling design variables, per phase.

    ``q`` is breadth (share of target-layer nodes carrying at least one
    incoming dependency); ``kappa`` is intensity (the fraction of a source's
    degradation that is transmitted along an edge). ``psi = q * kappa`` is the
    candidate sufficient statistic tested by H3.
    """

    q_prop: float = 0.0
    kappa_prop: float = 0.0
    q_rec: float = 0.0
    kappa_rec: float = 0.0
    placement_prop: str = "random"
    placement_rec: str = "random"

    @property
    def psi_prop(self) -> float:
        return float(self.q_prop * self.kappa_prop)

    @property
    def psi_rec(self) -> float:
        return float(self.q_rec * self.kappa_rec)

    def validate(self) -> None:
        for name in ("q_prop", "kappa_prop", "q_rec", "kappa_rec"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")


@dataclass(frozen=True)
class Scenario:
    """Complete description of one run."""

    window: HazardWindow = field(default_factory=HazardWindow)
    arm: Arm = field(default_factory=lambda: Arm(True, True))
    seeds: SeedBundle = field(default_factory=SeedBundle)
    design: CouplingDesign = field(default_factory=CouplingDesign)

    # Case-level hazard parameters, interpreted by the case's hazard field.
    hazard: Mapping[str, Any] = field(default_factory=dict)

    dt: float = 1.0
    t_end: float = 600.0
    resources_per_step: float = 1.0

    # Evaluation settings for the three headline metrics.
    T_eval: float = 400.0
    eta: float = 0.01
    eta_steps: int = 2

    tag: str = ""

    def __post_init__(self) -> None:
        self.window.validate()
        self.design.validate()
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.t_end <= self.window.t_repair_start:
            raise ValueError("t_end must extend beyond the repair embargo")

    # -- effective design given the arm ---------------------------------
    @property
    def effective_design(self) -> CouplingDesign:
        """Design with switched-off phases forced to zero coupling.

        Switching an arm off is exactly setting that phase's ``q`` and
        ``kappa`` to zero, so the four ablation arms are corner points of the
        same design space rather than a separate mechanism.
        """
        return CouplingDesign(
            q_prop=self.design.q_prop if self.arm.prop_on else 0.0,
            kappa_prop=self.design.kappa_prop if self.arm.prop_on else 0.0,
            q_rec=self.design.q_rec if self.arm.rec_on else 0.0,
            kappa_rec=self.design.kappa_rec if self.arm.rec_on else 0.0,
            placement_prop=self.design.placement_prop,
            placement_rec=self.design.placement_rec,
        )

    def for_arm(self, arm: Arm | str) -> "Scenario":
        """Return the same scenario under a different arm (seeds preserved)."""
        target = arm if isinstance(arm, Arm) else Arm.from_name(arm)
        return replace(self, arm=target)

    def for_replication(self, index: int) -> "Scenario":
        return replace(self, seeds=self.seeds.with_replication(index))

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data["hazard"] = _freeze(self.hazard)
        data["arm"] = self.arm.name
        return data

    def scenario_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def pairing_key(self) -> str:
        """Identity of the comparison this run belongs to.

        Two runs share a pairing key exactly when they differ only in the arm.
        Paired differences are only meaningful within one pairing key, and the
        factorial runner asserts this.
        """
        data = self.to_dict()
        data.pop("arm", None)
        data.pop("tag", None)
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _freeze(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_freeze(v) for v in value]
    return value
