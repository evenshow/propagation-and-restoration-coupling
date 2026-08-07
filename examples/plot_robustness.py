"""Figure 8: the two continuous axes along which the reported share could move.

Panel (a) answers what a different draw of coupling edges would do, which the
headline intervals cannot: they resample flood realisations against one fixed
layout. Panel (b) answers what a different evaluation horizon would do, which
matters because a locked-in run keeps accumulating loss while a recovered one
stops, so the share is an increasing function of the window rather than a
constant of the system.

Colour keeps its meaning from the rest of the paper -- aqua is the interaction --
so the two cases are separated by line style and marker fill rather than by hue.

    python coupling0729/examples/plot_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

INT = "#1baf7a"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
REPORTED = {1: 4000, 2: 7000}
HEADLINE = {1: 0.770, 2: 0.608}


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3, labelsize=8.5)


def shares_by_layout(path, n_boot=4000, seed=0):
    """Both layout estimands, kept distinct because they are not the same number.

    The dots are per-layout ratios of means; their unweighted average describes
    the cloud. The pooled ratio takes both means over every layout-hazard unit at
    once, which weights a layout by the size of its denominator, and is the
    estimand of Equation (9) applied to the wider population. Its interval comes
    from resampling whole layouts and recomputing the pooled ratio inside each
    resample -- resampling replications would not see layout variation at all.
    """
    d = pd.read_csv(path, dtype={"arm": str})
    d["arm"] = d["arm"].str.zfill(2)
    per_layout, units = [], {}
    for design_seed, g in d.groupby("design_seed"):
        w = g.pivot(index="replication", columns="arm", values="loss")
        total = w["11"] - w["00"]
        dint = total - (w["10"] - w["00"]) - (w["01"] - w["00"])
        per_layout.append(dint.mean() / total.mean())
        units[design_seed] = pd.DataFrame({"dint": dint, "total": total})

    keys = np.array(list(units))
    allu = pd.concat(units.values(), ignore_index=True)
    pooled = allu.dint.mean() / allu.total.mean()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = pd.concat([units[k] for k in keys[rng.integers(0, keys.size, keys.size)]],
                      ignore_index=True)
        boot.append(s.dint.mean() / s.total.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return np.sort(np.array(per_layout)), pooled, lo, hi


def shares_by_horizon(path):
    d = pd.read_csv(path, dtype={"arm": str})
    d["arm"] = d["arm"].str.zfill(2)
    cols = sorted([c for c in d.columns if c.startswith("loss_T")],
                  key=lambda c: int(c[6:]))
    ts, shares = [], []
    for c in cols:
        w = d.pivot(index="replication", columns="arm", values=c)
        total = w["11"] - w["00"]
        dint = total - (w["10"] - w["00"]) - (w["01"] - w["00"])
        ts.append(int(c[6:]))
        shares.append(dint.mean() / total.mean())
    return np.array(ts), np.array(shares)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="coupling0729/results")
    parser.add_argument("--out",
                        default="coupling0729/results/hpc_n1000/fig8_robustness.png")
    args = parser.parse_args()
    root = Path(args.indir)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    for case, offset, filled in ((1, -0.16, True), (2, 0.16, False)):
        s, pooled, lo, hi = shares_by_layout(root / f"case{case}_layouts" / "runs.csv")
        # The cloud and the two summary marks get their own columns; overlaying
        # them buries the interval in the dots it is meant to summarise.
        cloud, summary = case + offset - 0.10, case + offset + 0.15
        ax.scatter(cloud + np.linspace(-0.07, 0.07, s.size), s, s=34,
                   facecolor=INT if filled else SURFACE, edgecolor=INT,
                   linewidths=1.4, zorder=3)
        ax.errorbar([summary], [pooled], yerr=[[pooled - lo], [hi - pooled]],
                    fmt="D", ms=7, color=INK, ecolor=INK, elinewidth=1.6, capsize=5,
                    capthick=1.6, markeredgecolor=SURFACE, markeredgewidth=1.0,
                    zorder=5)
        ax.scatter([summary], [HEADLINE[case]], marker="_", s=520,
                   color="#eb6834", linewidths=2.4, zorder=4)
        ax.annotate(f"mean {s.mean():.3f}\nsd {s.std(ddof=1):.3f}", (cloud, 0.985),
                    ha="center", va="top", fontsize=8.2, color=MUTED,
                    linespacing=1.45)
        ax.annotate(f"{pooled:.3f}", (summary + 0.055, pooled), ha="left",
                    va="center", fontsize=8.2, color=INK)
    ax.set_xticks([1 - 0.14, 2 + 0.18])
    ax.set_xticklabels(["case 1\npower–transport", "case 2\nwater–transport"], fontsize=8.6)
    ax.set_xlim(0.58, 2.52)
    ax.set_ylim(0.0, 1.0)
    style(ax, "", "ρ_int", "(a)  across independent coupling layouts")

    ax = axes[1]
    for case, dashed, filled in ((1, False, True), (2, True, False)):
        t, s = shares_by_horizon(root / f"case{case}_multiT" / "runs.csv")
        ax.plot(t / REPORTED[case], s, color=INT, lw=2.0,
                ls="--" if dashed else "-", marker="o", ms=6,
                markerfacecolor=INT if filled else SURFACE,
                markeredgecolor=INT, markeredgewidth=1.4, zorder=3,
                label=f"case {case}")
    ax.axvline(1.0, color=AXIS, lw=1.0, ls=":", zorder=2)
    ax.annotate("reported\nhorizon", (1.0, 0.06), fontsize=8.2, color=MUTED,
                ha="center", linespacing=1.4)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=8.6, frameon=False)
    style(ax, "evaluation horizon,  multiples of the reported T", "ρ_int",
          "(b)  against the evaluation horizon")

    handles = [
        Line2D([], [], color=INT, lw=0, marker="o", ms=6, markerfacecolor=INT,
               label="one layout (ratio of means over its replications)"),
        Line2D([], [], color=INK, lw=1.6, marker="D", ms=6,
               label="pooled over all units, 95% CI clustered by layout"),
        Line2D([], [], color="#eb6834", lw=2.2, label="the layout reported in Table 2"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8.0, frameon=False,
                   labelspacing=0.4)

    fig.suptitle(
        "How far the reported interaction share moves along two axes the headline "
        "intervals do not cover\n"
        "(a) 20 independent coupling layouts, 100 replications each; "
        "(b) 300 replications re-evaluated over a ladder of horizons",
        fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
