"""Dose-response sweeps over the coupling design variables.

Two questions are asked here.

**H3 -- is ``psi = q * kappa`` a sufficient statistic?** If it is, the effect
surface over the ``(q, kappa)`` plane has *hyperbolic* iso-effect contours,
because ``q * kappa = const`` is a hyperbola. Design points that share a value
of ``psi`` but differ in how it is composed should then land on one curve. This
is a much sharper test than a regression coefficient, and it is the quantitative
form of the generality argument: if different cases collapse onto the same curve
in ``psi``, generality is a result rather than a claim about the software.

**H5 -- is there a containment threshold?** Above some level of buffering, does
extra propagation breadth stop mattering (``d Delta_prop / d q -> 0``)?

Both sweeps exploit the fact that **arm 00 does not depend on the coupling
design at all**: with the design loop inside the replication loop, one arm-00
run is reused across every design point of that replication, removing a quarter
of the work. ``test_arm_00_has_no_coupling_effect_at_all`` pins this property.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from ..core.scenario import ARM_NAMES, CouplingDesign, Scenario
from ..metrics.headline import MetricConfig, result_metrics
from .factorial import HEADLINE_METRICS, paired_effects

DESIGN_COLUMNS = ("q_prop", "kappa_prop", "q_rec", "kappa_rec", "seed_design")
COMPOSITION_COLUMNS = ("q_prop", "kappa_prop", "q_rec", "kappa_rec")


def psi_grid(
    q_values: Sequence[float],
    kappa_values: Sequence[float],
    couple_both_phases: bool = True,
    design_seeds: Sequence[int] | None = None,
) -> list[dict]:
    """Full ``q x kappa`` grid, applied to both phases simultaneously.

    Applying the same ``(q, kappa)`` to both phases is not a shortcut that
    confounds them: ``Delta_prop`` is measured on arm 10, where the restoration
    operator is switched off entirely, and ``Delta_rec`` on arm 01, where the
    propagation operator is off. Each main effect therefore depends on its own
    phase's design only, and one grid yields both dose-response surfaces.

    ``design_seeds`` replicates every ``(q, kappa)`` composition over several
    edge layouts. Without it each composition is a single placement draw, and
    within-iso-``psi`` spread cannot distinguish "``psi`` is not sufficient" from
    "this layout happened to differ", so the collapse ratio is only a lower
    bound. See :func:`placement_decomposition`.
    """
    seeds = list(design_seeds) if design_seeds is not None else [None]
    points = []
    for q in q_values:
        for kappa in kappa_values:
            if couple_both_phases:
                base = {"q_prop": float(q), "kappa_prop": float(kappa),
                        "q_rec": float(q), "kappa_rec": float(kappa)}
            else:
                base = {"q_prop": float(q), "kappa_prop": float(kappa),
                        "q_rec": 0.0, "kappa_rec": 0.0}
            for seed in seeds:
                point = dict(base)
                if seed is not None:
                    point["seed_design"] = int(seed)
                points.append(point)
    return points


def replication_rows(
    engine,
    base_scenario: Scenario,
    design_points: Sequence[dict],
    replication: int,
    arms: Sequence[str] = ARM_NAMES,
    indicator: str = "operational",
    weights: Mapping[str, float] | None = None,
    config: MetricConfig | None = None,
) -> list[dict]:
    """Every row for one hazard replication.

    Arm 00 is evaluated once and reused across all design points, which is valid
    because it depends on neither the coupling design nor the layout seed.
    """
    arms = list(arms)
    scenario = base_scenario.for_replication(replication)
    rows: list[dict] = []

    arm00: dict | None = None
    if "00" in arms:
        arm00 = result_metrics(
            engine.run(scenario.for_arm("00")),
            kind=indicator, weights=weights, config=config,
        )

    for point in design_points:
        # seed_design lives in the seed bundle, not the design record: it
        # selects the edge layout, not how much coupling there is.
        knobs = {k: v for k, v in point.items() if k != "seed_design"}
        design = replace(base_scenario.design, **knobs)
        scenario_d = replace(scenario, design=design)
        if "seed_design" in point:
            scenario_d = replace(
                scenario_d,
                seeds=replace(scenario_d.seeds, design=int(point["seed_design"])),
            )
        for arm in arms:
            if arm == "00":
                record = dict(arm00)
                record["arm"] = "00"
            else:
                result = engine.run(scenario_d.for_arm(arm))
                record = result_metrics(
                    result, kind=indicator, weights=weights, config=config
                )
                record.update(_realised(result.coupling_summary))
            record.update(point)
            record["replication"] = replication
            record["psi_prop"] = design.psi_prop
            record["psi_rec"] = design.psi_rec
            rows.append(record)
    return rows


def run_design_sweep(
    engine,
    base_scenario: Scenario,
    design_points: Sequence[dict],
    replications: int,
    arms: Sequence[str] = ARM_NAMES,
    indicator: str = "operational",
    weights: Mapping[str, float] | None = None,
    config: MetricConfig | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Run every design point under every requested arm, with shared randomness."""
    rows: list[dict] = []
    for replication in range(int(replications)):
        rows += replication_rows(
            engine, base_scenario, design_points, replication,
            arms=arms, indicator=indicator, weights=weights, config=config,
        )
        if progress:
            print(f"  replication {replication + 1}/{replications}", flush=True)
    return pd.DataFrame(rows)


def _parallel_block(payload: tuple) -> list[dict]:
    """Worker entry point: build one engine, then run a block of replications.

    Replications are handed out in contiguous blocks rather than round-robin so
    that each worker pays the engine construction cost once.
    """
    (factory, factory_kwargs, base_scenario, design_points,
     replications, arms, indicator, weights, config) = payload
    engine = factory(**dict(factory_kwargs or {}))
    rows: list[dict] = []
    for replication in replications:
        rows += replication_rows(
            engine, base_scenario, design_points, replication,
            arms=arms, indicator=indicator, weights=weights, config=config,
        )
    return rows


def run_design_sweep_parallel(
    engine_factory,
    base_scenario: Scenario,
    design_points: Sequence[dict],
    replications: int,
    workers: int | None = None,
    factory_kwargs: Mapping | None = None,
    arms: Sequence[str] = ARM_NAMES,
    indicator: str = "operational",
    weights: Mapping[str, float] | None = None,
    config: MetricConfig | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Same sweep, spread across processes.

    Replications are independent, so this is exact rather than approximate: each
    worker reproduces the identical common-random-number stream for the
    replications it owns, and the concatenated result does not depend on the
    worker count.

    ``engine_factory`` must be importable by name (a module-level function), and
    the caller must guard its entry point with ``if __name__ == "__main__"`` --
    both are process-spawn requirements on Windows.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    total = int(replications)
    workers = int(workers or min(os.cpu_count() or 1, total))
    workers = max(1, min(workers, total))

    def _sequential() -> pd.DataFrame:
        engine = engine_factory(**dict(factory_kwargs or {}))
        return run_design_sweep(
            engine, base_scenario, design_points, total,
            arms=arms, indicator=indicator, weights=weights,
            config=config, progress=progress,
        )

    if workers == 1:
        return _sequential()

    # Each worker's numpy/scipy would otherwise open a full thread pool of its
    # own, so W workers oversubscribe the machine W-fold. That is the classic
    # cause of workers being terminated abruptly on Windows, and even when it
    # does not kill them it costs throughput on an embarrassingly parallel
    # workload. Children inherit the environment at spawn, so this must be set
    # in the parent before the pool is created.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(variable, "1")

    blocks = [list(range(start, total, workers)) for start in range(workers)]
    blocks = [b for b in blocks if b]
    payloads = [
        (engine_factory, dict(factory_kwargs or {}), base_scenario, list(design_points),
         block, list(arms), indicator, weights, config)
        for block in blocks
    ]

    rows: list[dict] = []
    done = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for block_rows in pool.map(_parallel_block, payloads):
                rows += block_rows
                done += 1
                if progress:
                    print(f"  worker block {done}/{len(payloads)} done", flush=True)
    except BrokenProcessPool:
        # A long sweep must not be lost to a worker dying. Falling back is slow
        # but exact -- the sequential path reproduces the same random streams.
        print(
            "  [warn] worker pool broke; falling back to sequential execution",
            flush=True,
        )
        return _sequential()

    frame = pd.DataFrame(rows)
    # Restore a deterministic order so downstream output does not depend on the
    # order in which workers happened to finish.
    sort_keys = [c for c in ("replication", *DESIGN_COLUMNS, "arm") if c in frame.columns]
    return frame.sort_values(sort_keys).reset_index(drop=True)


def _realised(coupling_summary: dict) -> dict:
    """Pull the coupling actually built, which need not equal what was asked for.

    ``q`` is quantised by the target-layer size: a request of 0.6 on a 33-node
    layer becomes 20/33 = 0.606. That is harmless on its own, but it inflates
    within-iso-psi spread and therefore biases the H3 collapse test *against*
    sufficiency, so the realised values are recorded and should be used as the
    x-axis whenever they differ.
    """
    out = {}
    for phase, prefix in (("propagation", "prop"), ("restoration", "rec")):
        summary = coupling_summary.get(phase, {})
        out[f"q_{prefix}_realised"] = float(summary.get("q", 0.0))
        out[f"kappa_{prefix}_realised"] = float(summary.get("kappa_bar", 0.0))
        out[f"psi_{prefix}_realised"] = float(summary.get("psi", 0.0))
    return out


def quantisation_report(sweep: pd.DataFrame, phase: str = "prop") -> pd.DataFrame:
    """Requested vs realised breadth, for checking the H3 x-axis is honest."""
    q_col, realised = f"q_{phase}", f"q_{phase}_realised"
    if realised not in sweep.columns:
        return pd.DataFrame()
    active = sweep[sweep[realised] > 0.0]
    if active.empty:
        return pd.DataFrame()
    report = (
        active.groupby(q_col, as_index=False)[realised]
        .mean()
        .rename(columns={q_col: "requested_q", realised: "realised_q"})
    )
    report["abs_error"] = (report["realised_q"] - report["requested_q"]).abs()
    report["exact"] = report["abs_error"] < 1e-9
    return report


def sweep_effects(
    sweep: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
    design_columns: Sequence[str] = DESIGN_COLUMNS,
) -> pd.DataFrame:
    """Apply the paired differencing within each design point."""
    keys = [c for c in design_columns if c in sweep.columns]
    frames = []
    for values, group in sweep.groupby(keys, sort=True):
        effects = paired_effects(group, metrics=metrics)
        for key, value in zip(keys, values if isinstance(values, tuple) else (values,)):
            effects[key] = value
        frames.append(effects)
    out = pd.concat(frames, ignore_index=True)
    out["psi_prop"] = out["q_prop"] * out["kappa_prop"]
    out["psi_rec"] = out["q_rec"] * out["kappa_rec"]
    return out


def design_means(
    effects: pd.DataFrame,
    columns: Sequence[str],
    design_columns: Sequence[str] = DESIGN_COLUMNS,
) -> pd.DataFrame:
    """Average each effect within a design point, keeping the design variables."""
    keys = [c for c in design_columns if c in effects.columns]
    aggregated = effects.groupby(keys, as_index=False)[list(columns)].mean()
    aggregated["psi_prop"] = aggregated["q_prop"] * aggregated["kappa_prop"]
    aggregated["psi_rec"] = aggregated["q_rec"] * aggregated["kappa_rec"]
    return aggregated


# ---------------------------------------------------------------------------
# H3: does the surface collapse onto psi?
# ---------------------------------------------------------------------------
def collapse_diagnostic(
    aggregated: pd.DataFrame,
    column: str,
    psi_column: str = "psi_prop",
    tolerance: float = 1e-9,
) -> dict:
    """Compare within-iso-``psi`` spread against total spread.

    ``collapse_ratio = 1 - Var_within / Var_total`` over the design points that
    belong to an iso-``psi`` group with at least two members. A value near 1
    means design points sharing a ``psi`` agree with each other far better than
    the surface varies overall, i.e. ``psi`` carries essentially all the signal.
    Values near 0 mean ``q`` and ``kappa`` act separately and ``psi`` is not
    sufficient.
    """
    frame = aggregated[[psi_column, column]].dropna().copy()
    frame["psi_key"] = frame[psi_column].round(9)
    groups = frame.groupby("psi_key")[column]

    sizes = groups.size()
    shared = sizes[sizes >= 2].index
    if len(shared) == 0:
        return {
            "column": column,
            "n_iso_psi_groups": 0,
            "collapse_ratio": np.nan,
            "max_within_group_range": np.nan,
            "total_range": float(frame[column].max() - frame[column].min()),
        }

    subset = frame[frame["psi_key"].isin(shared)]
    within = subset.groupby("psi_key")[column].transform("mean")
    var_within = float(np.mean((subset[column] - within) ** 2))
    var_total = float(np.var(subset[column]))
    ranges = subset.groupby("psi_key")[column].agg(lambda s: float(s.max() - s.min()))

    return {
        "column": column,
        "n_iso_psi_groups": int(len(shared)),
        "n_points_in_groups": int(len(subset)),
        "collapse_ratio": float(1.0 - var_within / var_total) if var_total > tolerance else np.nan,
        "max_within_group_range": float(ranges.max()),
        "median_within_group_range": float(ranges.median()),
        "total_range": float(frame[column].max() - frame[column].min()),
    }


def placement_decomposition(
    per_seed: pd.DataFrame,
    column: str,
    psi_column: str = "psi_prop",
    composition_columns: Sequence[str] = COMPOSITION_COLUMNS,
) -> dict:
    """Separate placement variance from genuine insufficiency of ``psi``.

    The design is nested: edge layouts (``seed_design``) sit inside compositions
    ``(q, kappa)``, and compositions are grouped by ``psi``. The question "does
    composition matter beyond ``psi``?" is then a one-way comparison *within*
    each iso-``psi`` group, using layout-to-layout variation as the error term:

    * ``MS_within``  -- spread across layouts of the same composition;
    * ``MS_between`` -- spread across compositions sharing a ``psi``.

    Under H3 the composition contributes nothing, both mean squares have the
    same expectation, and ``F = MS_between / MS_within`` is 1. ``F`` well above
    1 means designs with equal ``psi`` differ by more than layout noise can
    explain, so ``psi`` is genuinely not sufficient.

    ``collapse_ratio_corrected`` removes the layout contribution from the
    numerator of the naive ratio and is therefore the estimate to quote; the
    naive ratio is a lower bound.
    """
    keys = [c for c in composition_columns if c in per_seed.columns]
    frame = per_seed[keys + [psi_column, column]].dropna().copy()
    frame["psi_key"] = frame[psi_column].round(9)

    ss_within = ss_between = 0.0
    df_within = df_between = 0
    seeds_per_composition: list[float] = []
    composition_means: list[float] = []
    groups_used = 0

    for _psi, group in frame.groupby("psi_key"):
        composition = group.groupby(keys)[column]
        means, sizes = composition.mean(), composition.size()
        if len(means) < 2 or bool((sizes < 2).any()):
            continue
        groups_used += 1
        weights = sizes.to_numpy(float)
        grand = float((means.to_numpy(float) * weights).sum() / weights.sum())
        for _key, values in composition:
            v = values.to_numpy(float)
            ss_within += float(((v - v.mean()) ** 2).sum())
            df_within += v.size - 1
        ss_between += float((weights * (means.to_numpy(float) - grand) ** 2).sum())
        df_between += len(means) - 1
        seeds_per_composition.append(float(weights.mean()))

    composition_means = (
        frame.groupby(keys)[column].mean().to_numpy(float) if len(frame) else np.array([])
    )
    total_variance = float(np.var(composition_means)) if composition_means.size else np.nan

    out = {
        "column": column,
        "n_iso_psi_groups": groups_used,
        "mean_layouts_per_composition": float(np.mean(seeds_per_composition))
        if seeds_per_composition else np.nan,
        "ms_within_placement": np.nan,
        "ms_between_composition": np.nan,
        "f_ratio": np.nan,
        "p_value": np.nan,
        "var_component_composition": np.nan,
        "collapse_ratio_corrected": np.nan,
        "total_variance": total_variance,
    }
    if groups_used == 0 or df_within == 0 or df_between == 0:
        return out

    ms_within = ss_within / df_within
    ms_between = ss_between / df_between
    n_bar = float(np.mean(seeds_per_composition))
    out["ms_within_placement"] = float(ms_within)
    out["ms_between_composition"] = float(ms_between)

    if ms_within > 0:
        f_ratio = ms_between / ms_within
        out["f_ratio"] = float(f_ratio)
        try:
            from scipy.stats import f as f_dist

            out["p_value"] = float(f_dist.sf(f_ratio, df_between, df_within))
        except Exception:
            pass

    # Variance attributable to composition once layout noise is removed.
    component = max(0.0, (ms_between - ms_within) / max(n_bar, 1e-9))
    out["var_component_composition"] = float(component)
    if np.isfinite(total_variance) and total_variance > 0:
        out["collapse_ratio_corrected"] = float(1.0 - component / total_variance)
    return out


def effective_coupling_fit(
    aggregated: pd.DataFrame,
    column: str,
    phase: str = "prop",
    kappa_0_grid: Sequence[float] | None = None,
) -> dict:
    """Fit the effective coupling ``psi_eff = q * (kappa - kappa_0)+``.

    ``psi = q * kappa`` treats breadth and intensity as perfect substitutes. A
    buffer breaks that symmetry: per-node degradation is roughly
    ``max(0, kappa*d - b)`` for a source deficit ``d`` and buffer allowance
    ``b``, so summing over the ``q`` fraction of coupled nodes gives

        effect ~ q * (kappa*d - b)  =  q*d * (kappa - kappa_0),   kappa_0 = b/d

    i.e. breadth times *excess* intensity. The extra ``-q*b`` term is why, at
    equal ``psi``, a design with high intensity and low breadth bites harder
    than one with low intensity and high breadth: each additional coupled node
    brings its own buffer allowance with it.

    ``kappa_0`` is not a free curve-fitting parameter -- it is the containment
    threshold of H5 read on the intensity axis, so the same quantity can be
    predicted from the marginal slopes and cross-checked against this fit.
    """
    q_col, k_col = f"q_{phase}", f"kappa_{phase}"
    frame = aggregated[[q_col, k_col, column]].dropna()
    q = frame[q_col].to_numpy(float)
    kappa = frame[k_col].to_numpy(float)
    y = frame[column].to_numpy(float)

    grid = (
        np.asarray(kappa_0_grid, dtype=float)
        if kappa_0_grid is not None
        else np.arange(0.0, float(kappa.max()), 0.005)
    )

    baseline_r2, baseline_slope = _r2_through_origin(q * kappa, y)
    best = {"kappa_0": 0.0, "r2": baseline_r2, "slope": baseline_slope}
    for kappa_0 in grid:
        r2, slope = _r2_through_origin(q * np.clip(kappa - kappa_0, 0.0, None), y)
        if r2 > best["r2"]:
            best = {"kappa_0": float(kappa_0), "r2": float(r2), "slope": float(slope)}

    return {
        "column": column,
        "phase": phase,
        "r2_psi": float(baseline_r2),
        "kappa_0": float(best["kappa_0"]),
        "r2_psi_eff": float(best["r2"]),
        "slope_psi_eff": float(best["slope"]),
        "r2_gain": float(best["r2"] - baseline_r2),
    }


def predicted_absorption_threshold(
    aggregated: pd.DataFrame, column: str, phase: str = "prop"
) -> dict:
    """Predict ``kappa_0`` from the marginal ``q``-slopes, independently of the fit.

    Under the buffer model ``d(effect)/dq = kappa*d - b``, so regressing the
    ``q``-slope on ``kappa`` recovers ``d`` and ``b`` and hence
    ``kappa_0 = b/d``. Agreement with :func:`effective_coupling_fit` is evidence
    that the functional form is right rather than merely flexible.
    """
    q_col, k_col = f"q_{phase}", f"kappa_{phase}"
    rows = []
    for kappa, group in aggregated.groupby(k_col):
        group = group.sort_values(q_col)
        if len(group) < 2:
            continue
        slope = float(
            np.polyfit(group[q_col].to_numpy(float), group[column].to_numpy(float), 1)[0]
        )
        rows.append((float(kappa), slope))
    if len(rows) < 2:
        return {"deficit_d": np.nan, "buffer_b": np.nan, "kappa_0_predicted": np.nan}

    kappa_values = np.array([r[0] for r in rows], dtype=float)
    slopes = np.array([r[1] for r in rows], dtype=float)
    # Fit only where the effect has broken through, otherwise the clipped zeros
    # drag the line and the recovered threshold is meaningless.
    active = slopes > 1e-9
    if active.sum() < 2:
        return {"deficit_d": np.nan, "buffer_b": np.nan, "kappa_0_predicted": np.nan}
    d, minus_b = np.polyfit(kappa_values[active], slopes[active], 1)
    b = -float(minus_b)
    return {
        "deficit_d": float(d),
        "buffer_b": b,
        "kappa_0_predicted": float(b / d) if abs(d) > 1e-12 else np.nan,
    }


def _r2_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denominator = float((x * x).sum())
    if denominator <= 1e-15:
        return (-np.inf, np.nan)
    slope = float((x * y).sum() / denominator)
    residual = y - slope * x
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 1e-15:
        return (np.nan, slope)
    return (float(1.0 - (residual**2).sum() / total), slope)


def marginal_slopes(aggregated: pd.DataFrame, column: str, phase: str = "prop") -> pd.DataFrame:
    """Slope of the effect in ``q`` at each fixed ``kappa``, and vice versa.

    Useful as a companion to the collapse ratio: under ``psi`` sufficiency the
    slope in ``q`` must scale with ``kappa`` and the slope in ``kappa`` with
    ``q``.
    """
    q_col, k_col = f"q_{phase}", f"kappa_{phase}"
    rows = []
    for held, varying, label in ((k_col, q_col, "d/dq at fixed kappa"),
                                 (q_col, k_col, "d/dkappa at fixed q")):
        for value, group in aggregated.groupby(held):
            group = group.sort_values(varying)
            if len(group) < 2:
                continue
            slope = float(
                np.polyfit(group[varying].to_numpy(float), group[column].to_numpy(float), 1)[0]
            )
            rows.append({"kind": label, "held_at": float(value), "slope": slope})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H5: containment threshold
# ---------------------------------------------------------------------------
def containment_threshold(
    curves: pd.DataFrame,
    control_column: str,
    q_column: str = "q_prop",
    effect_column: str = "Lambda__delta_prop",
    flatten_fraction: float = 0.2,
) -> pd.DataFrame:
    """Slope of the effect in ``q`` at each level of a control variable.

    The threshold is the smallest control level whose ``q``-slope has fallen to
    ``flatten_fraction`` of the slope at the weakest control level: beyond it,
    extra propagation breadth buys little extra damage.
    """
    rows = []
    for value, group in curves.groupby(control_column):
        group = group.sort_values(q_column)
        if len(group) < 2:
            continue
        slope = float(
            np.polyfit(group[q_column].to_numpy(float), group[effect_column].to_numpy(float), 1)[0]
        )
        rows.append({control_column: float(value), "q_slope": slope, "n_points": len(group)})

    frame = pd.DataFrame(rows).sort_values(control_column).reset_index(drop=True)
    if frame.empty:
        return frame
    reference = float(frame.loc[0, "q_slope"])
    frame["slope_ratio"] = frame["q_slope"] / reference if abs(reference) > 1e-12 else np.nan
    frame["flattened"] = frame["slope_ratio"].abs() <= float(flatten_fraction)
    return frame
