"""Tests for the attribution table's additivity guarantees.

The additive identity holds *within* a replication by construction. It survives
averaging only if every channel is averaged over the same replications, which
requires the metric to be defined in every arm. ``T_rec`` is not — it does not
exist when a run never recovers — so these tests pin the machinery that makes
that visible instead of silently producing a row that does not sum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.experiments import (
    attribution_audit,
    attribution_table,
    conditional_recovery_table,
)
from coupling0729.experiments.attribution import NOT_ADDITIVE_MARK

METRICS = ("loss", "T_rec")


def _effects(censor: int = 0) -> pd.DataFrame:
    """Synthetic effects: ``loss`` always defined, ``T_rec`` censored in the
    interaction/total channels for the last ``censor`` replications."""
    n = 12
    rng = np.random.default_rng(3)
    prop = rng.uniform(0.01, 0.03, n)
    rec = rng.uniform(0.02, 0.05, n)
    inter = rng.uniform(0.00, 0.04, n)

    frame = pd.DataFrame(
        {
            "replication": np.arange(n),
            "loss__delta_prop": prop,
            "loss__delta_rec": rec,
            "loss__delta_int": inter,
            "loss__total": prop + rec + inter,
            "T_rec__delta_prop": prop * 1000,
            "T_rec__delta_rec": rec * 1000,
            "T_rec__delta_int": inter * 1000,
            "T_rec__total": (prop + rec + inter) * 1000,
        }
    )
    if censor:
        # Arm 11 never recovered in these replications, so anything needing it
        # is undefined -- exactly the real censoring pattern.
        frame.loc[frame.index[-censor:], ["T_rec__delta_int", "T_rec__total"]] = np.nan
    return frame


def test_uncensored_rows_sum_exactly():
    table = attribution_table(_effects(censor=0), metrics=METRICS)
    for label, row in table.iterrows():
        assert NOT_ADDITIVE_MARK not in label
        parts = row.iloc[0] + row.iloc[1] + row.iloc[2]
        assert parts == pytest.approx(row.iloc[3], abs=1e-12)


def test_censored_row_is_marked_not_additive():
    table = attribution_table(_effects(censor=5), metrics=METRICS)
    labels = list(table.index)
    marked = [label for label in labels if NOT_ADDITIVE_MARK in label]
    assert len(marked) == 1 and "T_rec" in marked[0]
    assert all(NOT_ADDITIVE_MARK not in label for label in labels if "T_rec" not in label)


def test_marked_row_really_does_not_sum():
    """The mark is not cosmetic: the arithmetic genuinely fails."""
    table = attribution_table(_effects(censor=5), metrics=METRICS)
    row = table.loc[[i for i in table.index if NOT_ADDITIVE_MARK in i][0]]
    parts = row.iloc[0] + row.iloc[1] + row.iloc[2]
    assert abs(parts - row.iloc[3]) > 1e-6


def test_complete_case_restores_additivity():
    table = attribution_table(_effects(censor=5), metrics=METRICS, complete_case=True)
    for label, row in table.iterrows():
        assert NOT_ADDITIVE_MARK not in label, "complete case must equalise the samples"
        parts = row.iloc[0] + row.iloc[1] + row.iloc[2]
        assert parts == pytest.approx(row.iloc[3], abs=1e-12)


def test_audit_reports_the_sample_sizes_and_the_residual():
    audit = attribution_audit(_effects(censor=5), metrics=METRICS).set_index("metric")

    assert bool(audit.loc["loss", "samples_consistent"])
    assert bool(audit.loc["loss", "additive"])
    assert audit.loc["loss", "residual"] == pytest.approx(0.0, abs=1e-12)

    assert not bool(audit.loc["T_rec", "samples_consistent"])
    assert not bool(audit.loc["T_rec", "additive"])
    assert audit.loc["T_rec", "n_delta_prop"] == 12
    assert audit.loc["T_rec", "n_delta_int"] == 7
    assert audit.loc["T_rec", "n_complete_case"] == 7
    assert abs(audit.loc["T_rec", "residual"]) > 1e-6


def test_partial_censoring_does_not_trip_the_additivity_assertion():
    """Regression: the check summed the three channels with pandas' default
    skipna, which treats NaN as zero. A replication censored in arm 01 only --
    delta_rec and delta_int undefined, total finite -- then compared `total`
    against `delta_prop` alone and raised on an identity that held. It killed a
    1000-replication merge after the simulation had already been paid for."""
    from coupling0729.experiments import paired_effects

    n = 8
    runs = []
    for replication in range(n):
        for arm, value in (("00", 100.0), ("10", 110.0), ("01", 160.0), ("11", 200.0)):
            censored = replication >= 5 and arm == "01"   # arm 01 only
            runs.append(
                {
                    "replication": replication,
                    "arm": arm,
                    "loss": 0.1 + 0.01 * len(arm.strip("0")),
                    "T_rec": np.nan if censored else value,
                    "T_rec_censored": censored,
                    "T_rec_bound": 900.0,
                }
            )
    frame = pd.DataFrame(runs)

    effects = paired_effects(frame, metrics=("T_rec",))   # must not raise
    ok = effects.dropna(subset=["T_rec__delta_int"])
    assert len(ok) == 5, "the five uncensored replications remain checkable"
    parts = ok[["T_rec__delta_prop", "T_rec__delta_rec", "T_rec__delta_int"]].sum(axis=1)
    np.testing.assert_allclose(parts, ok["T_rec__total"], atol=1e-12)
    assert effects["T_rec__delta_rec"].isna().sum() == 3


def test_conditional_recovery_table_has_no_total():
    table = conditional_recovery_table(_effects(censor=5), metric="T_rec")
    assert "total" not in [str(c).lower() for c in table["channel"]]
    assert set(table.columns) == {"channel", "n", "n_censored", "mean", "median"}
    interaction = table[table["channel"].str.contains("interaction")].iloc[0]
    assert interaction["n"] == 7 and interaction["n_censored"] == 5
