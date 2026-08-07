"""The 2x2 factorial with common random numbers.

For every hazard replication the same four runs are executed::

    arm 00   both operators off        (independent networks)
    arm 10   propagation only
    arm 01   restoration only
    arm 11   both

All four share ``seed_hazard`` and ``seed_sim``, so they see byte-identical
physical damage. Any difference between arms is therefore attributable to the
coupling operators alone, and the three effects

    Delta_prop = X10 - X00
    Delta_rec  = X01 - X00
    Delta_int  = X11 - X10 - X01 + X00

satisfy ``X11 - X00 = Delta_prop + Delta_rec + Delta_int`` **exactly** -- an
algebraic identity, not an approximation, which is what makes the attribution
table add up.

``Delta_int`` is the term the existing literature cannot form, because it
requires the two phase-specific dependencies to be switchable independently.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..core.scenario import ARM_NAMES, Scenario
from ..metrics.headline import MetricConfig, result_metrics

HEADLINE_METRICS = ("loss", "Lambda", "T_rec")


def run_factorial(
    engine,
    base_scenario: Scenario,
    replications: int,
    indicator: str = "operational",
    weights: dict | None = None,
    config: MetricConfig | None = None,
    results_root=None,
    keep_trajectories: bool = False,
    progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run ``replications`` x 4 arms and return tidy per-run metrics."""
    rows: list[dict] = []
    trajectories: dict[tuple[int, str], object] = {}

    for replication in range(int(replications)):
        scenario = base_scenario.for_replication(replication)
        pairing = {scenario.for_arm(a).pairing_key() for a in ARM_NAMES}
        if len(pairing) != 1:
            raise AssertionError(
                "the four arms of a replication must differ only in the arm field; "
                "common random numbers are broken"
            )

        for arm in ARM_NAMES:
            run_scenario = scenario.for_arm(arm)
            result = engine.run(run_scenario, results_root=results_root)
            record = result_metrics(result, kind=indicator, weights=weights, config=config)
            record.update(
                {
                    "replication": replication,
                    "seed_hazard": run_scenario.seeds.hazard,
                    "q_prop": run_scenario.design.q_prop,
                    "kappa_prop": run_scenario.design.kappa_prop,
                    "q_rec": run_scenario.design.q_rec,
                    "kappa_rec": run_scenario.design.kappa_rec,
                    "psi_prop": run_scenario.design.psi_prop,
                    "psi_rec": run_scenario.design.psi_rec,
                    "frac_damaged": result.damage_summary.get("frac_damaged", np.nan),
                }
            )
            # Realised coupling can differ from what was requested: q is
            # quantised by the target-layer size (0.5 on a 33-bus feeder becomes
            # 16/33 = 0.485). Record it so the design axis reported is the one
            # actually simulated.
            record.update(_realised_coupling(result.coupling_summary))
            rows.append(record)
            if keep_trajectories:
                trajectories[(replication, arm)] = result

        if progress:
            print(f"  replication {replication + 1}/{replications} done")

    frame = pd.DataFrame(rows)
    return frame, trajectories


def _realised_coupling(coupling_summary: dict) -> dict:
    """Breadth and intensity as actually built, per phase."""
    out = {}
    for phase, prefix in (("propagation", "prop"), ("restoration", "rec")):
        summary = coupling_summary.get(phase, {})
        out[f"q_{prefix}_realised"] = float(summary.get("q", 0.0))
        out[f"kappa_{prefix}_realised"] = float(summary.get("kappa_bar", 0.0))
    return out


def paired_effects(
    runs: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Form the three paired differences per replication.

    Censored ``T_rec`` values propagate as NaN. A ``*_bounded`` companion is
    also produced in which censored runs are replaced by the censoring bound;
    that variant is a *conservative under-estimate* of the effect and must be
    reported as such, never as the headline number.
    """
    frames = []
    for metric in metrics:
        wide = runs.pivot(index="replication", columns="arm", values=metric)
        effects = _effects_from_wide(wide, metric)

        if metric == "T_rec" and "T_rec_bound" in runs.columns:
            bound = runs.groupby("replication")["T_rec_bound"].max()
            filled = wide.apply(lambda col: col.fillna(bound))
            effects = effects.join(
                _effects_from_wide(filled, f"{metric}_bounded"), how="left"
            )
        frames.append(effects)

    out = pd.concat(frames, axis=1)
    censored = runs.groupby("replication")["T_rec_censored"].any()
    out["any_censored"] = censored.reindex(out.index).fillna(False)
    return out.reset_index()


def _effects_from_wide(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Form whichever paired differences the available arms support.

    A reduced sweep that only needs one main effect (arms 00 and 10, say) is a
    legitimate design and costs half as much, so this computes what it can
    rather than demanding the full factorial. The interaction and the total
    require all four arms by definition.
    """
    available = set(wide.columns)
    if "00" not in available:
        raise ValueError(f"arm 00 is required to form any paired difference for {metric!r}")

    frame = pd.DataFrame(index=wide.index)
    if "10" in available:
        frame[f"{metric}__delta_prop"] = wide["10"] - wide["00"]
    if "01" in available:
        frame[f"{metric}__delta_rec"] = wide["01"] - wide["00"]

    if available >= set(ARM_NAMES):
        total = wide["11"] - wide["00"]
        frame[f"{metric}__total"] = total
        frame[f"{metric}__delta_int"] = (
            total - frame[f"{metric}__delta_prop"] - frame[f"{metric}__delta_rec"]
        )
        # min_count=3 makes the sum NaN unless all three channels are present.
        # Without it pandas skips NaN as if it were zero, so a replication in
        # which only arm 01 is censored -- delta_rec and delta_int undefined but
        # total finite -- compares `total` against `delta_prop` alone and the
        # check fires on an identity that was never violated. The identity is
        # only checkable where every channel is defined.
        parts = frame[
            [f"{metric}__delta_prop", f"{metric}__delta_rec", f"{metric}__delta_int"]
        ].sum(axis=1, min_count=3)
        residual = total - parts
        finite = residual[np.isfinite(residual)]
        worst = float(np.max(np.abs(finite))) if len(finite) else 0.0
        if worst > 1e-9:
            raise AssertionError(
                f"additive identity violated for {metric!r} (max residual {worst:.3g})"
            )
    elif not frame.columns.size:
        raise ValueError(f"no usable arm pair present for {metric!r}: got {sorted(available)}")
    return frame


def effect_summary(
    effects: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Median, mean and bootstrap CI of each effect, plus the stopping diagnostic.

    ``ci_halfwidth`` is the quantity to watch when deciding how many
    replications are enough: because the arms share random numbers, it shrinks
    far faster than the spread of the levels themselves.
    """
    rows = []
    for metric in metrics:
        for channel in ("delta_prop", "delta_rec", "delta_int", "total"):
            column = f"{metric}__{channel}"
            if column not in effects.columns:
                continue
            values = effects[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            low, high = _bootstrap_ci(finite, confidence)
            rows.append(
                {
                    "metric": metric,
                    "channel": channel,
                    "n": int(finite.size),
                    "n_missing": int(values.size - finite.size),
                    "mean": float(np.mean(finite)) if finite.size else np.nan,
                    "median": float(np.median(finite)) if finite.size else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_halfwidth": (high - low) / 2 if np.isfinite(low) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_ci(
    values: np.ndarray, confidence: float, draws: int = 4000, seed: int = 12345
) -> tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))
