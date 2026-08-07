"""Tests for the dose-response sweep layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.toy2layer import base_scenario, build_case
from coupling0729.experiments import (
    collapse_diagnostic,
    containment_threshold,
    design_means,
    psi_grid,
    run_design_sweep,
    sweep_effects,
)

LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]


def test_psi_grid_contains_iso_psi_groups():
    """The collapse test is only possible if some psi values are reachable
    from more than one (q, kappa) composition."""
    points = psi_grid(LEVELS, LEVELS)
    assert len(points) == 25
    psi = pd.Series([round(p["q_prop"] * p["kappa_prop"], 9) for p in points])
    shared = psi.value_counts()
    assert (shared >= 2).sum() >= 8, "grid must admit several iso-psi groups"
    assert shared.max() >= 3


def _synthetic_surface(fn) -> pd.DataFrame:
    rows = []
    for q in LEVELS:
        for k in LEVELS:
            rows.append({"q_prop": q, "kappa_prop": k, "q_rec": q, "kappa_rec": k,
                         "effect": fn(q, k)})
    frame = pd.DataFrame(rows)
    frame["psi_prop"] = frame["q_prop"] * frame["kappa_prop"]
    frame["psi_rec"] = frame["psi_prop"]
    return frame


def test_collapse_ratio_is_one_when_psi_is_sufficient():
    surface = _synthetic_surface(lambda q, k: 3.0 * (q * k) ** 1.5)
    out = collapse_diagnostic(surface, "effect", "psi_prop")
    assert out["collapse_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert out["max_within_group_range"] == pytest.approx(0.0, abs=1e-9)


def test_collapse_ratio_falls_when_psi_is_not_sufficient():
    # Depends on breadth alone: equal-psi designs then disagree strongly.
    surface = _synthetic_surface(lambda q, k: 2.0 * q)
    out = collapse_diagnostic(surface, "effect", "psi_prop")
    assert out["collapse_ratio"] < 0.5
    assert out["max_within_group_range"] > 0.1


def test_collapse_diagnostic_handles_absent_groups():
    surface = _synthetic_surface(lambda q, k: q * k)
    single = surface.drop_duplicates(subset="psi_prop")  # no psi reachable twice
    out = collapse_diagnostic(single, "effect", "psi_prop")
    assert out["n_iso_psi_groups"] == 0
    assert np.isnan(out["collapse_ratio"])


def test_arm_00_is_reused_and_identical_across_design_points():
    """The sweep's central optimisation: arm 00 ignores the coupling design."""
    engine = build_case(n_supply=8, n_service=8)
    points = psi_grid([0.4, 0.8], [0.4, 0.8])
    sweep = run_design_sweep(engine, base_scenario(), points, replications=2)

    for replication, group in sweep[sweep["arm"] == "00"].groupby("replication"):
        assert group["loss"].nunique() == 1
        assert group["Lambda"].nunique() == 1
        assert len(group) == len(points)


def test_two_arm_sweep_yields_only_the_propagation_effect():
    engine = build_case(n_supply=8, n_service=8)
    points = psi_grid([0.4, 0.8], [0.8])
    sweep = run_design_sweep(
        engine, base_scenario(), points, replications=2, arms=("00", "10")
    )
    effects = sweep_effects(sweep, metrics=("Lambda",))
    assert "Lambda__delta_prop" in effects.columns
    assert "Lambda__delta_rec" not in effects.columns
    assert "Lambda__delta_int" not in effects.columns


def test_sweep_effects_keeps_design_columns():
    engine = build_case(n_supply=8, n_service=8)
    points = psi_grid([0.4, 0.8], [0.8])
    sweep = run_design_sweep(engine, base_scenario(), points, replications=2)
    aggregated = design_means(
        sweep_effects(sweep, metrics=("Lambda",)), ["Lambda__delta_prop"]
    )
    assert set(aggregated["q_prop"]) == {0.4, 0.8}
    assert "psi_prop" in aggregated.columns


def test_propagation_effect_increases_with_breadth():
    engine = build_case(n_supply=12, n_service=12)
    points = psi_grid([0.2, 1.0], [0.8])
    sweep = run_design_sweep(
        engine, base_scenario(), points, replications=3, arms=("00", "10")
    )
    aggregated = design_means(
        sweep_effects(sweep, metrics=("Lambda",)), ["Lambda__delta_prop"]
    ).sort_values("q_prop")
    depths = aggregated["Lambda__delta_prop"].to_numpy(float)
    assert depths[-1] > depths[0], "wider propagation coupling must deepen the drop"


def _per_seed_surface(fn, n_seeds: int = 4, noise: float = 0.004, seed: int = 5) -> pd.DataFrame:
    """Synthetic per-layout effects: one row per (q, kappa, seed_design)."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in LEVELS:
        for k in LEVELS:
            for s in range(n_seeds):
                rows.append(
                    {
                        "q_prop": q, "kappa_prop": k, "q_rec": q, "kappa_rec": k,
                        "seed_design": s,
                        "effect": fn(q, k) + rng.normal(0.0, noise),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["psi_prop"] = frame["q_prop"] * frame["kappa_prop"]
    return frame


def test_placement_decomposition_finds_no_composition_effect_when_psi_suffices():
    from coupling0729.experiments import placement_decomposition

    per_seed = _per_seed_surface(lambda q, k: 3.0 * (q * k))
    out = placement_decomposition(per_seed, "effect", "psi_prop")
    assert out["n_iso_psi_groups"] >= 8
    assert out["mean_layouts_per_composition"] == pytest.approx(4.0)
    # Composition adds nothing beyond layout noise: F near 1, not significant.
    assert 0.2 < out["f_ratio"] < 4.0
    assert out["p_value"] > 0.01
    assert out["collapse_ratio_corrected"] > 0.98


def test_placement_decomposition_detects_genuine_insufficiency():
    from coupling0729.experiments import placement_decomposition

    # Depends on breadth alone, so equal-psi compositions differ systematically.
    per_seed = _per_seed_surface(lambda q, k: 2.0 * q)
    out = placement_decomposition(per_seed, "effect", "psi_prop")
    assert out["f_ratio"] > 20.0
    assert out["p_value"] < 1e-6
    assert out["collapse_ratio_corrected"] < 0.6


def test_placement_decomposition_needs_repeated_layouts():
    from coupling0729.experiments import placement_decomposition

    single = _per_seed_surface(lambda q, k: q * k, n_seeds=1)
    out = placement_decomposition(single, "effect", "psi_prop")
    assert out["n_iso_psi_groups"] == 0
    assert np.isnan(out["f_ratio"])


def test_psi_grid_expands_over_design_seeds():
    points = psi_grid(LEVELS, LEVELS, design_seeds=[0, 1, 2])
    assert len(points) == 75
    assert all("seed_design" in p for p in points)
    assert {p["seed_design"] for p in points} == {0, 1, 2}


def test_design_seed_changes_the_layout_not_the_dose():
    """seed_design must move the edge placement while leaving q and kappa alone."""
    engine = build_case(n_supply=12, n_service=12)
    points = psi_grid([0.5], [0.8], design_seeds=[0, 1, 2, 3])
    sweep = run_design_sweep(
        engine, base_scenario(), points, replications=2, arms=("00", "10")
    )
    active = sweep[sweep["arm"] == "10"]
    assert active["q_prop_realised"].nunique() == 1, "breadth must not vary with the seed"
    assert active["kappa_prop_realised"].nunique() == 1
    # Different layouts of the same dose generally give different outcomes.
    assert active.groupby("seed_design")["Lambda"].mean().nunique() > 1


def test_parallel_sweep_reproduces_the_sequential_one():
    """Parallelism must be exact, not approximate.

    Each worker regenerates the same common-random-number stream for the
    replications it owns, so the concatenated frame must not depend on how many
    workers were used. If this ever fails, every paired difference is suspect.
    """
    from coupling0729.cases.toy2layer import build_case as toy_factory
    from coupling0729.experiments import run_design_sweep_parallel

    points = psi_grid([0.4, 0.8], [0.6])
    kwargs = {"n_supply": 8, "n_service": 8}
    scenario = base_scenario()

    sequential = run_design_sweep(
        toy_factory(**kwargs), scenario, points, replications=4, arms=("00", "10")
    )
    parallel = run_design_sweep_parallel(
        toy_factory, scenario, points, replications=4, workers=2,
        factory_kwargs=kwargs, arms=("00", "10"),
    )

    keys = ["replication", "q_prop", "kappa_prop", "arm"]
    left = sequential.sort_values(keys).reset_index(drop=True)
    right = parallel.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left[keys + ["loss", "Lambda"]], right[keys + ["loss", "Lambda"]]
    )


def test_effective_coupling_recovers_a_known_threshold():
    """A surface built as q*(kappa-0.25)+ must be recovered as such."""
    from coupling0729.experiments import effective_coupling_fit

    surface = _synthetic_surface(lambda q, k: 0.5 * q * max(0.0, k - 0.25))
    surface = surface.rename(columns={"effect": "y"})
    out = effective_coupling_fit(surface, "y", "prop")
    assert out["kappa_0"] == pytest.approx(0.25, abs=0.02)
    assert out["r2_psi_eff"] > 0.999
    assert out["r2_psi_eff"] > out["r2_psi"], "excess-intensity form must beat q*kappa"
    assert out["slope_psi_eff"] == pytest.approx(0.5, rel=0.05)


def test_effective_coupling_keeps_psi_when_psi_is_right():
    """If the truth really is q*kappa, the fit must not invent a threshold."""
    from coupling0729.experiments import effective_coupling_fit

    surface = _synthetic_surface(lambda q, k: 2.0 * q * k)
    surface = surface.rename(columns={"effect": "y"})
    out = effective_coupling_fit(surface, "y", "prop")
    assert out["kappa_0"] == pytest.approx(0.0, abs=0.02)
    assert out["r2_gain"] == pytest.approx(0.0, abs=1e-6)


def test_absorption_threshold_prediction_matches_the_fit():
    """The slope-based prediction is an independent route to the same kappa_0."""
    from coupling0729.experiments import (
        effective_coupling_fit,
        predicted_absorption_threshold,
    )

    surface = _synthetic_surface(lambda q, k: 0.5 * q * max(0.0, k - 0.25))
    surface = surface.rename(columns={"effect": "y"})
    fitted = effective_coupling_fit(surface, "y", "prop")["kappa_0"]
    predicted = predicted_absorption_threshold(surface, "y", "prop")["kappa_0_predicted"]
    assert predicted == pytest.approx(fitted, abs=0.05)


def test_containment_threshold_detects_flattening():
    curves = pd.concat(
        [
            pd.DataFrame(
                {
                    "buffer": buffer,
                    "q_prop": LEVELS,
                    # slope decays with buffer: 0.2 -> 0.2/(1+buffer)
                    "Lambda__delta_prop": [0.2 / (1.0 + buffer) * q for q in LEVELS],
                }
            )
            for buffer in (0.0, 1.0, 4.0, 9.0, 19.0)
        ],
        ignore_index=True,
    )
    out = containment_threshold(curves, "buffer", flatten_fraction=0.2)
    assert out.loc[0, "slope_ratio"] == pytest.approx(1.0)
    flattened = out[out["flattened"]]
    assert not flattened.empty
    assert float(flattened["buffer"].iloc[0]) == pytest.approx(4.0)
