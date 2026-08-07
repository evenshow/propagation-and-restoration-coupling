"""How much of the coupling-induced effect the interaction carries, as a
function of the design point.

The headline result is reported at one design point (``q = 0.5``, ``kappa =
0.8``), which leaves open whether interaction dominance is a property of that
point or of the system. This module reduces each design point to one number,

    rho_int(q, kappa) = E[Delta_int] / E[Delta_total]

so the question becomes a surface rather than an anecdote.

Two estimator choices carry the result, and both are the kind that quietly
decide an answer if left to default:

**A ratio of means, never a mean of ratios.** Per replication the denominator is
``X11 - X00``, the coupling-induced loss for that particular flood. For a mild
realisation it is near zero, and nothing forbids it from being *negative*, so
per-replication shares are unbounded and change sign. Averaging them yields a
number dominated by whichever replications happened to have the smallest
denominators. The ratio of means is the share of the *aggregate* effect, which
is what the paper already reports at the baseline ("77% of the coupling-induced
loss"), and it is well behaved as long as the mean total is bounded away from
zero.

**A paired bootstrap.** Replications are resampled once per draw and both means
are recomputed from that same resample, preserving the strong positive
correlation between numerator and denominator. Resampling them independently
would inflate the interval substantially.

The share is meaningless where the total is itself indistinguishable from zero
-- at weak coupling there is no effect to take a share of -- so
:func:`interaction_share` reports whether the denominator's own interval
excludes zero, and :func:`share_surface` refuses to quote a share where it does
not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DESIGN_KEYS = ("q_prop", "kappa_prop", "q_rec", "kappa_rec")


def interaction_share(
    effects: pd.DataFrame,
    metric: str = "loss",
    confidence: float = 0.95,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict:
    """``E[Delta_int] / E[Delta_total]`` with a paired bootstrap interval."""
    numer = effects[f"{metric}__delta_int"].to_numpy(dtype=float)
    denom = effects[f"{metric}__total"].to_numpy(dtype=float)
    keep = np.isfinite(numer) & np.isfinite(denom)
    numer, denom = numer[keep], denom[keep]
    n = numer.size
    if n == 0:
        raise ValueError(f"no replications with both channels finite for {metric!r}")

    mean_int, mean_total = float(numer.mean()), float(denom.mean())
    share = mean_int / mean_total if abs(mean_total) > 1e-12 else float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_int = numer[idx].mean(axis=1)
    boot_total = denom[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_share = np.where(np.abs(boot_total) > 1e-12, boot_int / boot_total, np.nan)

    alpha = (1.0 - confidence) / 2.0
    finite_share = boot_share[np.isfinite(boot_share)]
    lo, hi = (np.quantile(finite_share, [alpha, 1 - alpha])
              if finite_share.size else (float("nan"), float("nan")))
    total_lo, total_hi = np.quantile(boot_total, [alpha, 1 - alpha])

    return {
        "metric": metric,
        "n": int(n),
        "mean_int": mean_int,
        "mean_total": mean_total,
        "total_ci_low": float(total_lo),
        "total_ci_high": float(total_hi),
        # Below this the design point has no aggregate effect to take a share
        # of, and rho is a ratio of two numbers that are both noise.
        "total_is_positive": bool(total_lo > 0.0),
        "share": float(share),
        "share_ci_low": float(lo),
        "share_ci_high": float(hi),
    }


def share_surface(
    effects: pd.DataFrame,
    metric: str = "loss",
    design_keys: tuple[str, ...] = DESIGN_KEYS,
    confidence: float = 0.95,
    n_boot: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per design point: the share, its interval, and the raw levels.

    The absolute effects travel with the share on purpose. A design point can
    show ``rho_int = 0.9`` because the interaction dominates a substantial loss,
    or because it dominates a loss too small to care about; the share alone
    cannot tell those apart, and reporting it alone would invite the reader to
    treat them as the same finding.
    """
    keys = [k for k in design_keys if k in effects.columns]
    if not keys:
        raise ValueError(f"none of {design_keys} present in the effects frame")

    rows = []
    for values, group in effects.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        record = dict(zip(keys, (float(v) for v in values)))
        stats = interaction_share(group, metric=metric, confidence=confidence,
                                  n_boot=n_boot, seed=seed)
        if not stats["total_is_positive"]:
            stats["share"] = float("nan")
            stats["share_ci_low"] = float("nan")
            stats["share_ci_high"] = float("nan")
        for channel in ("delta_prop", "delta_rec", "delta_int"):
            column = f"{metric}__{channel}"
            record[f"mean_{channel}"] = float(group[column].mean(skipna=True))
        record.update(stats)
        rows.append(record)

    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
