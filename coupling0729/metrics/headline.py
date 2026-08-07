"""The three headline resilience metrics.

Deliberately three, with distinct roles rather than three parallel answers:

======================  ==========================================  ===========
metric                  definition                                  role
======================  ==========================================  ===========
``loss = 1 - R``        ``R = (1/T) int_{t_oe}^{t_oe+T} F(t) dt``    outcome
``Lambda``              ``1 - min F`` over ``[t_oe, t_ee + t_mob]``  mediator: depth
``T_rec``               ``T_eta - t_ee``                            mediator: delay
======================  ==========================================  ===========

Two definitional choices carry the design and must not be silently changed:

1. ``Lambda`` is read over the repair-embargo window only. No restoration can
   occur there, so the depth metric cannot absorb restoration-coupling effects
   and the phase-separation hypothesis is testable rather than tautological.
2. ``T_rec`` is anchored at ``t_ee`` -- exogenous, identical in every arm --
   and *not* at the observed onset of recovery. Anchoring at the observed onset
   would subtract out exactly the delay that restoration coupling causes.

All three are reported in the loss direction (larger is worse) so that signs in
the attribution table are never ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.scenario import HazardWindow
from ..core.trajectory import TrajectoryResult


@dataclass(frozen=True)
class MetricConfig:
    """Evaluation settings shared by every run in a comparison."""

    T_eval: float = 400.0
    eta: float = 0.01
    eta_steps: int = 2

    def horizon_end(self, window: HazardWindow) -> float:
        return float(window.t_oe + self.T_eval)

    def censoring_bound(self, window: HazardWindow) -> float:
        """Largest ``T_rec`` that the evaluation horizon could have observed."""
        return float(self.horizon_end(window) - window.t_ee)


def curve_metrics(
    curve: pd.Series,
    window: HazardWindow,
    config: MetricConfig | None = None,
) -> dict:
    """Compute the three headline metrics from one normalised performance curve."""
    config = config or MetricConfig()
    series = curve.astype(float).clip(upper=1.0)
    horizon_end = config.horizon_end(window)

    if float(series.index[-1]) + 1e-9 < horizon_end:
        raise ValueError(
            f"trajectory ends at t={float(series.index[-1])} but the evaluation "
            f"horizon needs t={horizon_end}; increase Scenario.t_end or lower T_eval"
        )

    evaluation = series.loc[(series.index >= window.t_oe) & (series.index <= horizon_end)]
    r_value = _mean_on_grid(evaluation)

    embargo = series.loc[
        (series.index >= window.t_oe) & (series.index <= window.t_repair_start)
    ]
    f_min = float(embargo.min())
    lambda_value = float(np.clip(1.0 - f_min, 0.0, 1.0))
    t_min = float(embargo.idxmin())

    t_eta = _recovery_instant(evaluation, window.t_ee, config.eta, config.eta_steps)
    censored = t_eta is None

    return {
        "R": float(r_value),
        "loss": float(1.0 - r_value),
        "Lambda": lambda_value,
        "F_min": f_min,
        "t_min": t_min,
        "T_rec": float("nan") if censored else float(t_eta - window.t_ee),
        "T_rec_censored": bool(censored),
        "T_rec_bound": config.censoring_bound(window),
        # Loss-direction indicator of failure to recover within the horizon.
        # Reported separately because censoring is not a nuisance here: mutual
        # gating between the two phases can make recovery impossible, and that
        # lock-in is itself attributable through the same additive identity.
        "lock_in": 1.0 if censored else 0.0,
        "F_end": float(evaluation.iloc[-1]),
    }


def result_metrics(
    result: TrajectoryResult,
    kind: str = "operational",
    weights: dict | None = None,
    config: MetricConfig | None = None,
) -> dict:
    """Headline metrics for the system-level curve of one run."""
    config = config or MetricConfig(
        T_eval=result.scenario.T_eval,
        eta=result.scenario.eta,
        eta_steps=result.scenario.eta_steps,
    )
    curve = result.system_curve(kind=kind, weights=weights)
    out = curve_metrics(curve, result.scenario.window, config)
    out["arm"] = result.arm
    out["indicator"] = kind
    return out


def _mean_on_grid(series: pd.Series) -> float:
    """Time-average of a curve, using the trapezoid rule on its own grid."""
    if len(series) < 2:
        return float(series.iloc[0]) if len(series) else float("nan")
    x = series.index.to_numpy(dtype=float)
    y = series.to_numpy(dtype=float)
    integrate = getattr(np, "trapezoid", np.trapz)
    span = float(x[-1] - x[0])
    if span <= 0:
        return float(y[0])
    return float(integrate(y, x=x) / span)


def _recovery_instant(
    series: pd.Series, t_from: float, eta: float, steps: int
) -> float | None:
    """First instant at or after ``t_from`` where ``F >= 1 - eta`` holds ``steps`` times."""
    segment = series.loc[series.index >= t_from]
    run = 0
    first_index: float | None = None
    for t, value in segment.items():
        if float(value) >= 1.0 - float(eta):
            if run == 0:
                first_index = float(t)
            run += 1
            if run >= int(steps):
                return first_index
        else:
            run = 0
            first_index = None
    return None
