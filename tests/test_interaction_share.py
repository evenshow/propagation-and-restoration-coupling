"""Tests for the interaction-share estimator.

``rho_int`` carries a paper claim, and the three ways it can be silently wrong
all produce a plausible-looking number rather than an error:

* averaging per-replication ratios instead of taking the ratio of the means,
  which is dominated by the replications whose denominator happens to be near
  zero;
* bootstrapping numerator and denominator independently, which discards the
  correlation between them and inflates the interval;
* quoting a share at a design point whose total effect is itself
  indistinguishable from zero, where the ratio is two noise terms divided.

Each test below is built so that a correct implementation and the corresponding
mistake give *different* answers, not merely different precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.experiments import interaction_share, share_surface


def _effects(delta_int, total, **design) -> pd.DataFrame:
    """An effects frame carrying only what the estimator reads."""
    delta_int = np.asarray(delta_int, dtype=float)
    total = np.asarray(total, dtype=float)
    frame = pd.DataFrame({
        "loss__delta_int": delta_int,
        "loss__total": total,
        # The prop/rec split is arbitrary here; only their sum is constrained.
        "loss__delta_prop": (total - delta_int) / 2.0,
        "loss__delta_rec": (total - delta_int) / 2.0,
    })
    for key, value in design.items():
        frame[key] = value
    return frame


def test_share_is_a_ratio_of_means_not_a_mean_of_ratios():
    """The two estimators must disagree here, and we must get the former.

    One replication has a denominator a ten-thousandth the size of the other's.
    Its per-replication share is 100; the other's is 0.01. Averaging those gives
    ~50 -- a "share" far outside [0, 1], produced entirely by one small
    denominator. The ratio of means is 2 / 100.01.
    """
    effects = _effects(delta_int=[1.0, 1.0], total=[100.0, 0.01])

    stats = interaction_share(effects, n_boot=200, seed=1)

    assert stats["share"] == pytest.approx(2.0 / 100.01, rel=1e-9)
    mean_of_ratios = np.mean([1.0 / 100.0, 1.0 / 0.01])
    assert not np.isclose(stats["share"], mean_of_ratios, rtol=1e-3)


def test_bootstrap_is_paired():
    """Perfectly proportional channels must give a zero-width interval.

    If ``delta_int = c * total`` in every replication, then every resample --
    however extreme -- has that same ratio, so a paired bootstrap returns
    exactly ``c`` each draw. Resampling the two independently would pair a high
    numerator draw with a low denominator draw and produce a visibly wide band.
    """
    rng = np.random.default_rng(0)
    total = rng.uniform(0.01, 5.0, size=200)      # deliberately wide spread
    effects = _effects(delta_int=0.3 * total, total=total)

    stats = interaction_share(effects, n_boot=500, seed=2)

    assert stats["share"] == pytest.approx(0.3, rel=1e-12)
    assert stats["share_ci_low"] == pytest.approx(0.3, rel=1e-9)
    assert stats["share_ci_high"] == pytest.approx(0.3, rel=1e-9)
    assert stats["share_ci_high"] - stats["share_ci_low"] < 1e-9


def test_share_is_withheld_where_the_total_is_not_positive():
    """No aggregate effect means no share to report.

    The totals here are symmetric about zero, so the bootstrap interval on the
    mean total straddles it. ``interaction_share`` flags that; ``share_surface``
    acts on the flag and emits NaN rather than a ratio of two noise terms.
    """
    rng = np.random.default_rng(3)
    total = rng.normal(0.0, 1.0, size=300)
    effects = _effects(delta_int=rng.normal(0.0, 0.1, size=300), total=total,
                       q_prop=0.2, kappa_prop=0.2, q_rec=0.2, kappa_rec=0.2)

    stats = interaction_share(effects, n_boot=500, seed=4)
    assert stats["total_is_positive"] is False
    assert np.isfinite(stats["share"])            # the raw ratio still exists

    surface = share_surface(effects, n_boot=500, seed=4)
    row = surface.iloc[0]
    assert np.isnan(row["share"])
    assert np.isnan(row["share_ci_low"])
    assert np.isnan(row["share_ci_high"])
    # The absolute levels must survive: they are what tells the reader whether a
    # withheld share means "no effect" or "effect present but unshareable".
    assert np.isfinite(row["mean_int"])
    assert np.isfinite(row["mean_total"])
    assert np.isfinite(row["mean_delta_prop"])


def test_positive_total_keeps_the_share():
    """The complement of the previous test, so the guard cannot pass vacuously."""
    rng = np.random.default_rng(5)
    total = rng.normal(2.0, 0.1, size=300)
    effects = _effects(delta_int=0.6 * total, total=total,
                       q_prop=0.5, kappa_prop=0.8, q_rec=0.5, kappa_rec=0.8)

    surface = share_surface(effects, n_boot=500, seed=6)

    assert surface.iloc[0]["share"] == pytest.approx(0.6, rel=1e-9)
    assert bool(interaction_share(effects, n_boot=500, seed=6)["total_is_positive"])


def test_surface_returns_one_row_per_design_point():
    frames = []
    for q, kappa, ratio in ((0.5, 0.4, 0.2), (0.5, 0.8, 0.7), (1.0, 0.8, 0.9)):
        total = np.full(50, 1.0)
        frames.append(_effects(delta_int=ratio * total, total=total,
                               q_prop=q, kappa_prop=kappa, q_rec=q, kappa_rec=kappa))
    effects = pd.concat(frames, ignore_index=True)

    surface = share_surface(effects, n_boot=200, seed=7)

    assert len(surface) == 3
    assert surface["share"].tolist() == pytest.approx([0.2, 0.7, 0.9], rel=1e-9)
    assert surface["q_prop"].tolist() == [0.5, 0.5, 1.0]


def test_replications_missing_a_channel_are_dropped_from_both_means():
    """A NaN in either channel must remove the whole replication.

    Keeping it in one mean and not the other would make the numerator and
    denominator rest on different samples -- the same defect that keeps `T_rec`
    out of the additive table.
    """
    effects = _effects(delta_int=[1.0, np.nan, 3.0], total=[2.0, 5.0, 6.0])

    stats = interaction_share(effects, n_boot=200, seed=8)

    assert stats["n"] == 2
    assert stats["share"] == pytest.approx(4.0 / 8.0, rel=1e-12)


def test_empty_input_is_an_error_not_a_nan():
    effects = _effects(delta_int=[np.nan, np.nan], total=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        interaction_share(effects)
