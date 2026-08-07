"""Figure 6: where in the design space the interaction dominates.

Two rows, because the finding has two halves and one is evidence for the other.

The top row is the headline: ``rho_int``, the share of the coupling-induced loss
carried by the interaction, against coupling intensity, one line per breadth.
Drawn as lines rather than a heat map because what matters is a *threshold* --
a near-vertical rise between ``kappa = 0.6`` and ``0.8`` -- and a colour ramp
renders a step and a steep gradient almost identically.

The bottom row is why the top row is not simply "more coupling, more of
everything": at the same design points, both main effects cross the threshold
essentially unchanged while the interaction jumps by an order of magnitude. A
log axis is used because the three channels differ by three orders of magnitude
at weak coupling; it is labelled as such.

``q`` is an ordered quantity, so it gets a sequential ramp in a hue that carries
no mechanism meaning elsewhere in the paper. Blue, orange and aqua stay reserved
for propagation, restoration and interaction, and are used with exactly that
meaning in the bottom row.

    python coupling0729/examples/plot_rho_surface.py
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

PROP, REC, INT = "#2a78d6", "#eb6834", "#1baf7a"
# Sequential ramp for breadth q: ordered, and deliberately not a mechanism hue.
Q_RAMP = ("#cfc4e8", "#a88fd0", "#7f5cb4", "#583a8e", "#341f5c")
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
BAND = "#f0e9d8"
BASELINE_Q, BASELINE_K = 0.50, 0.80


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


def share_panel(ax, frame, title):
    ax.axvspan(0.6, 0.8, color=BAND, lw=0, zorder=0)
    q_levels = sorted(frame.q_prop.unique())
    for colour, q in zip(Q_RAMP, q_levels):
        sub = frame[frame.q_prop == q].sort_values("kappa_prop")
        x = sub.kappa_prop.to_numpy()
        y = sub.share.to_numpy()
        ax.fill_between(x, sub.share_ci_low, sub.share_ci_high,
                        color=colour, alpha=0.16, lw=0, zorder=1)
        # The lines converge above the threshold, so end labels would overlap;
        # breadth is carried by a legend instead.
        ax.plot(x, y, color=colour, lw=2.0, marker="o", ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3,
                label=f"q = {q:.2f}")

    base = frame[(frame.q_prop == BASELINE_Q) & (frame.kappa_prop == BASELINE_K)]
    if not base.empty:
        ax.plot([BASELINE_K], [base.iloc[0].share], marker="o", ms=13, mfc="none",
                markeredgecolor=INK, markeredgewidth=1.5, zorder=4)
        ax.annotate("design point used\nin Tables 1–3",
                    (BASELINE_K, base.iloc[0].share), textcoords="offset points",
                    xytext=(40, -34), ha="left", fontsize=8.0, color=INK,
                    linespacing=1.5, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))

    ax.text(0.7, 1.03, "transition interval", ha="center", fontsize=8.2,
            color="#8a6d00", weight="bold")
    legend = ax.legend(loc="upper left", fontsize=8.0, frameon=False,
                       title="breadth q", labelspacing=0.35, handlelength=1.6)
    legend.get_title().set_fontsize(8.2)
    legend.get_title().set_color(MUTED)
    ax.set_xlim(0.13, 1.12)
    ax.set_ylim(-0.03, 1.09)
    style(ax, "coupling intensity  κ  (both phases)",
          "ρ_int  =  E[Δ_int] / E[Δ_total]", title)


def channel_panel(ax, frame, title):
    ax.axvspan(0.6, 0.8, color=BAND, lw=0, zorder=0)
    sub = frame[frame.q_prop == BASELINE_Q].sort_values("kappa_prop")
    x = sub.kappa_prop.to_numpy()
    for column, colour, label in (("mean_delta_prop", PROP, "Δ_prop"),
                                  ("mean_delta_rec", REC, "Δ_rec"),
                                  ("mean_delta_int", INT, "Δ_int")):
        # Delta_prop and Delta_rec nearly coincide in case 1, so these rely on
        # the shared legend below the figure rather than on end labels.
        y = sub[column].to_numpy()
        ax.plot(x, y, color=colour, lw=2.0, marker="o", ms=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3, label=label)
    ax.set_yscale("log")
    ax.set_xlim(0.13, 1.07)
    style(ax, "coupling intensity  κ  (both phases)",
          "effect on 1 − R   (log scale)", title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="coupling0729/results/hpc_n1000")
    parser.add_argument("--out",
                        default="coupling0729/results/hpc_n1000/fig6_rho_surface.png")
    args = parser.parse_args()

    indir = Path(args.indir)
    frames = {}
    for case in (1, 2):
        path = indir / f"case{case}_rho_sweep" / "rho_int_surface.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        frames[case] = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    fig.patch.set_facecolor(SURFACE)

    share_panel(axes[0, 0], frames[1], "(a)  Case 1 — power and transport")
    share_panel(axes[0, 1], frames[2], "(b)  Case 2 — water and transport")
    channel_panel(axes[1, 0], frames[1], "(c)  Case 1 — the three channels at q = 0.50")
    channel_panel(axes[1, 1], frames[2], "(d)  Case 2 — the three channels at q = 0.50")

    handles = [
        Line2D([], [], color=PROP, lw=2.0, marker="o", ms=6, markeredgecolor=SURFACE,
               label="Δ_prop   propagation"),
        Line2D([], [], color=REC, lw=2.0, marker="o", ms=6, markeredgecolor=SURFACE,
               label="Δ_rec   restoration"),
        Line2D([], [], color=INT, lw=2.0, marker="o", ms=6, markeredgecolor=SURFACE,
               label="Δ_int   interaction"),
        Line2D([], [], color=Q_RAMP[2], lw=6, alpha=0.3,
               label="shaded band in (a)–(b): 95% bootstrap CI on ρ_int"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.015), columnspacing=2.2)

    fig.suptitle(
        "The attribution across the coupling design space\n"
        "200 hazard replications per design point; shaded band marks where the transition falls, "
        "bracketed by the grid step",
        fontsize=12, y=0.975)
    fig.tight_layout(rect=(0, 0.055, 1, 0.925))
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
