"""Hazard realisation: intensity field, fragility, and time-resolved damage.

The framework commits to one chain only::

    hazard realisation w  ->  intensity IM_i(w)  ->  fragility  ->  damage state

and deliberately not to any particular hazard. Earthquake (PGA), windstorm
(gust speed) and pluvial flooding (depth-duration) differ only in which
:class:`HazardField` and which fragility parameters a case plugs in.

Two properties matter for the identification strategy:

1. **One resistance draw per component.** Component ``i`` draws a single
   uniform ``u_i`` once, from the hazard seed. Its damage state at any time is
   a deterministic function of ``u_i`` and the intensity experienced so far.
   The same ``u_i`` is therefore reused across all four experimental arms,
   which is what makes the paired differences common-random-number estimates.
2. **Monotone accrual.** Damage is driven by the running maximum of the
   intensity, so a component never heals during the event and Phase I has a
   genuine slope instead of a single cliff.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .scenario import HazardWindow


class DamageState(IntEnum):
    """Ordinal damage states, following the usual HAZUS-style ladder."""

    NONE = 0
    SLIGHT = 1
    MODERATE = 2
    EXTENSIVE = 3
    COMPLETE = 4


N_DAMAGE_STATES = len(DamageState)


# ---------------------------------------------------------------------------
# Fragility
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LognormalFragility:
    """Standard lognormal fragility: ``P[DS >= s | IM] = Phi(ln(IM/lam_s)/beta_s)``.

    ``medians`` holds ``lam_s`` for s = SLIGHT..COMPLETE (four entries) and must
    be non-decreasing. ``betas`` holds the matching log standard deviations.
    """

    medians: tuple[float, ...]
    betas: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.medians) != N_DAMAGE_STATES - 1:
            raise ValueError(f"expected {N_DAMAGE_STATES - 1} medians")
        if len(self.betas) != len(self.medians):
            raise ValueError("medians and betas must have equal length")
        if any(b <= 0 for b in self.betas):
            raise ValueError("betas must be positive")
        if list(self.medians) != sorted(self.medians):
            raise ValueError("medians must be non-decreasing across damage states")

    def exceedance(self, intensity: float) -> np.ndarray:
        """Return ``P[DS >= s]`` for s = SLIGHT..COMPLETE."""
        im = float(intensity)
        if im <= 0.0:
            return np.zeros(len(self.medians))
        out = np.empty(len(self.medians))
        for idx, (lam, beta) in enumerate(zip(self.medians, self.betas)):
            out[idx] = _std_normal_cdf(math.log(im / float(lam)) / float(beta))
        # Exceedance probabilities must be non-increasing in the state index.
        return np.minimum.accumulate(out)

    def state_for(self, intensity: float, resistance: float) -> DamageState:
        """Map a uniform resistance draw to a damage state at this intensity."""
        exceed = self.exceedance(intensity)
        state = int(np.count_nonzero(exceed > float(resistance)))
        return DamageState(min(state, N_DAMAGE_STATES - 1))


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Intensity fields
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RadialHazardField:
    """Spatially correlated intensity field decaying from an epicentre.

    The mean field is ``peak * exp(-d / attenuation)``; a spatially correlated
    lognormal residual with exponential covariance (range ``corr_length``,
    log-sd ``sigma``) is added so that nearby components fail together. That
    correlation is the whole point: independent per-component failure would
    make cross-layer coupling effects look artificially small.
    """

    peak: float = 1.0
    attenuation: float = 4.0
    sigma: float = 0.4
    corr_length: float = 2.0
    epicentre: tuple[float, float] = (0.0, 0.0)

    def intensities(
        self,
        positions: Mapping[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        ids = list(positions)
        if not ids:
            return {}
        xy = np.asarray([positions[c] for c in ids], dtype=float)
        centre = np.asarray(self.epicentre, dtype=float)
        dist = np.linalg.norm(xy - centre, axis=1)
        mean = float(self.peak) * np.exp(-dist / max(float(self.attenuation), 1e-9))

        residual = _correlated_normal(xy, float(self.corr_length), rng)
        values = mean * np.exp(float(self.sigma) * residual)
        return {cid: float(max(v, 0.0)) for cid, v in zip(ids, values)}


def _correlated_normal(
    xy: np.ndarray, corr_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw a zero-mean unit-variance field with exponential covariance."""
    n = xy.shape[0]
    if n == 1:
        return rng.standard_normal(1)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    cov = np.exp(-d / max(float(corr_length), 1e-9))
    cov[np.diag_indices(n)] = 1.0 + 1e-8
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        chol = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
    return chol @ rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Temporal accrual
# ---------------------------------------------------------------------------
def intensity_factor(t: float, window: HazardWindow, shape: str = "ramp") -> float:
    """Running-maximum intensity factor in ``[0, 1]``, non-decreasing in ``t``.

    ``ramp``      linear build-up across the whole window;
    ``rise_peak`` linear build-up to the mid-window peak, flat thereafter.
    """
    if t < window.t_oe:
        return 0.0
    if t >= window.t_ee:
        return 1.0
    frac = (float(t) - window.t_oe) / window.duration
    if shape == "ramp":
        return float(frac)
    if shape == "rise_peak":
        return float(min(1.0, 2.0 * frac))
    raise ValueError(f"unknown hazard shape {shape!r}")


# ---------------------------------------------------------------------------
# Damage realisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DamageRealisation:
    """One hazard realisation ``w``, evaluated lazily in time.

    Holds the per-component intensity and the single uniform resistance draw.
    Damage at any time is then deterministic, which is what lets the four arms
    share exactly the same physical damage history.
    """

    intensity: Mapping[str, float]
    resistance: Mapping[str, float]
    fragility: Mapping[str, LognormalFragility]
    component_class: Mapping[str, str]
    window: HazardWindow
    shape: str = "ramp"
    seed_hazard: int = 0

    def state_at(self, component: str, t: float) -> DamageState:
        factor = intensity_factor(t, self.window, self.shape)
        if factor <= 0.0:
            return DamageState.NONE
        model = self.fragility[self.component_class[component]]
        return model.state_for(self.intensity[component] * factor, self.resistance[component])

    def final_state(self, component: str) -> DamageState:
        return self.state_at(component, self.window.t_ee)

    def final_states(self) -> dict[str, DamageState]:
        return {c: self.final_state(c) for c in self.intensity}

    def summary(self) -> dict:
        states = self.final_states()
        counts = {s.name: 0 for s in DamageState}
        for state in states.values():
            counts[state.name] += 1
        n = max(len(states), 1)
        return {
            "seed_hazard": int(self.seed_hazard),
            "n_components": len(states),
            "mean_intensity": float(np.mean(list(self.intensity.values()))) if states else 0.0,
            "frac_damaged": float(
                sum(1 for s in states.values() if s > DamageState.NONE) / n
            ),
            **{f"n_{k.lower()}": v for k, v in counts.items()},
        }


def sample_damage(
    positions: Mapping[str, tuple[float, float]],
    component_class: Mapping[str, str],
    fragility: Mapping[str, LognormalFragility],
    field_model: RadialHazardField,
    window: HazardWindow,
    seed_hazard: int,
    shape: str = "ramp",
) -> DamageRealisation:
    """Draw one hazard realisation.

    The RNG is seeded solely by ``seed_hazard``, so calling this with the same
    seed reproduces byte-identical damage regardless of the experimental arm.
    """
    unclassified = set(positions) - set(component_class)
    if unclassified:
        raise KeyError(
            "sample_damage was given positions without a fragility class: "
            f"{sorted(unclassified)[:5]}. Pass only damageable components; "
            "service nodes carry positions for coupling but take no damage."
        )
    rng = np.random.default_rng(int(seed_hazard))
    intensity = field_model.intensities(positions, rng)
    resistance = {cid: float(rng.random()) for cid in positions}
    missing = {component_class[c] for c in positions} - set(fragility)
    if missing:
        raise KeyError(f"no fragility model for component classes: {sorted(missing)}")
    return DamageRealisation(
        intensity=intensity,
        resistance=resistance,
        fragility=dict(fragility),
        component_class=dict(component_class),
        window=window,
        shape=shape,
        seed_hazard=int(seed_hazard),
    )
