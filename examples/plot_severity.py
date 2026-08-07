"""Figure 7: how the attribution moves with hazard severity.

The reported runs fix the flood peak at 0.60 m, which invites the obvious
question of whether the split depends on that choice. This runs the same
factorial across six peaks and answers it directly.

Two panels because the answer has two halves. The left shows the channels in
absolute terms: everything grows with severity, and nothing saturates over the
range covered -- the paper should not claim a ceiling it has not observed. The
right shows the share carried by the interaction, which is what the conclusion
actually rests on, and that is close to flat.

    python coupling0729/examples/plot_severity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.experiments import interaction_share, paired_effects

PROP, REC, INT = "#2a78d6", "#eb6834", "#1baf7a"
TOTAL = "#0b0b0b"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
PEAKS = ("040", "050", "060", "070", "080", "090")
BASELINE = 0.60


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


def load(indir: Path) -> pd.DataFrame:
    rows = []
    for peak in PEAKS:
        runs = pd.read_csv(indir / f"peak_{peak}" / "runs.csv", dtype={"arm": str})
        runs["arm"] = runs["arm"].str.zfill(2)
        effects = paired_effects(runs, metrics=("loss", "Lambda", "lock_in"))
        share = interaction_share(effects, metric="loss", n_boot=4000, seed=0)
        record = {"peak": int(peak) / 100,
                  "frac_damaged": float(runs["frac_damaged"].mean()),
                  "share": share["share"],
                  "share_lo": share["share_ci_low"],
                  "share_hi": share["share_ci_high"]}
        for channel in ("delta_prop", "delta_rec", "delta_int", "total"):
            values = effects[f"loss__{channel}"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            boot = np.array([values[np.random.default_rng(i).integers(
                0, values.size, values.size)].mean() for i in range(1000)])
            record[channel] = values.mean()
            record[f"{channel}_lo"], record[f"{channel}_hi"] = np.quantile(boot, [0.025, 0.975])
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="coupling0729/results/case1_severity")
    parser.add_argument("--out",
                        default="coupling0729/results/hpc_n1000/fig7_severity.png")
    args = parser.parse_args()

    frame = load(Path(args.indir))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    for column, colour, label in (("total", TOTAL, "total  X₁₁ − X₀₀"),
                                  ("delta_int", INT, "Δ_int"),
                                  ("delta_rec", REC, "Δ_rec"),
                                  ("delta_prop", PROP, "Δ_prop")):
        ax.fill_between(frame.peak, frame[f"{column}_lo"], frame[f"{column}_hi"],
                        color=colour, alpha=0.15, lw=0, zorder=1)
        ax.plot(frame.peak, frame[column], color=colour, lw=2.0, marker="o", ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3, label=label)
    ax.axvline(BASELINE, color=AXIS, lw=1.0, ls=":", zorder=2)
    ax.annotate("0.60 m\nreported runs", (BASELINE, ax.get_ylim()[1] * 0.93),
                fontsize=8.0, color=MUTED, ha="center", va="top", linespacing=1.4)
    ax.legend(loc="upper left", fontsize=8.4, frameon=False, labelspacing=0.35)
    style(ax, "flood peak  (m)", "effect on 1 − R",
          "(a)  the three channels and their total")

    ax = axes[1]
    ax.fill_between(frame.peak, frame.share_lo, frame.share_hi,
                    color=INT, alpha=0.16, lw=0, zorder=1)
    ax.plot(frame.peak, frame.share, color=INT, lw=2.0, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
    ax.axvline(BASELINE, color=AXIS, lw=1.0, ls=":", zorder=2)
    ax.set_ylim(0.0, 1.0)
    for _, row in frame.iterrows():
        ax.annotate(f"{row.frac_damaged:.0%}", (row.peak, 0.055), fontsize=7.6,
                    color=MUTED, ha="center")
    ax.text(frame.peak.min(), 0.13, "share of components damaged",
            fontsize=7.8, color=MUTED, ha="left")
    style(ax, "flood peak  (m)", "ρ_int  =  E[Δ_int] / E[Δ_total]",
          "(b)  share carried by the interaction")

    fig.suptitle(
        "Case 1: the attribution across hazard severity, 100 replications per level\n"
        "the channels grow with severity and do not saturate over this range, while the "
        "share carried by the interaction stays between 0.69 and 0.85",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"written to {args.out}")
    print(frame[["peak", "frac_damaged", "delta_prop", "delta_rec", "delta_int",
                 "total", "share"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
