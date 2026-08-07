"""Figure 5: attribution forest plot.

Three design rules carried through the whole figure:

* **Colour encodes the mechanism**, never the case and never the gate setting.
  Propagation is blue, restoration orange, interaction aqua, in that fixed order.
  Cases are separated by position, gate settings by filled/hollow markers.
* **Only metrics that decompose exactly appear here.** ``1-R``, ``Lambda`` and
  lock-in do, so their panels annotate the total. ``T_rec`` does not — it is
  undefined when a run never recovers, so its three channels rest on subsamples
  of different sizes. It had its own panel here and that was a mistake: the
  interaction channel is computed on the runs in which arm 11 recovered at all,
  which are the milder floods, and in case 2 that selection turned a real delay
  into an apparent 8-minute *speed-up* — a negative bar pointing the opposite way
  to the paper's finding, with no way to label the artefact out of it. Recovery
  is now shown where it needs no conditioning: as an empirical cumulative recovery curve
  over every replication in Figures 3 and 4, with the conditional means tabulated alongside
  their sample sizes.
* **One scale per panel.** Loss, depth, lock-in probability and minutes are four
  different units; they are four panels, never twinned axes.

    .venv_coupling/Scripts/python.exe coupling0729/examples/plot_attribution.py
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

# Categorical slots 1-3 of the reference palette; validated all-pairs in light
# mode (worst CVD dE 9.2, normal-vision 24.0).
CHANNEL_COLOUR = {
    "delta_prop": "#2a78d6",
    "delta_rec": "#eb6834",
    "delta_int": "#1baf7a",
}
CHANNEL_LABEL = {
    "delta_prop": "propagation",
    "delta_rec": "restoration",
    "delta_int": "interaction",
}
CHANNELS = ("delta_prop", "delta_rec", "delta_int")

# Below this, a value is a floating-point residue of an exact zero, not a small
# effect. Printing 4.7e-12 would read as one.
ZERO_TOL = 1e-9


def fmt(value: float) -> str:
    return "0" if abs(value) < ZERO_TOL else f"{value:.4g}"

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def load(indir: Path) -> dict[str, pd.DataFrame]:
    # The gate ablation was first run at 400 replications; those directories are
    # kept but no longer read, so every number in the figure rests on the same
    # 1000 and the caption needs no per-panel qualifier.
    sets = {
        "case1": "case1_n1000",
        "case2": "case2_n1000",
        "case1_gate": "case1_gate_infra_n1000",
        "case2_gate": "case2_gate_infra_n1000",
    }
    out = {}
    for key, folder in sets.items():
        path = indir / folder / "effect_summary.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        out[key] = pd.read_csv(path)
    return out


def pick(frame: pd.DataFrame, metric: str, channel: str) -> pd.Series:
    row = frame[(frame.metric == metric) & (frame.channel == channel)]
    if row.empty:
        raise KeyError(f"{metric}/{channel} not in summary")
    return row.iloc[0]


def forest_panel(ax, data, metric, title, xlabel, show_total=True):
    """Two case blocks, three mechanism rows each, newest at the top."""
    rows, ticks, labels = [], [], []
    y = 0.0
    for case_key, case_name in (("case2", "Case 2  water–transport"),
                                ("case1", "Case 1  power–transport")):
        frame = data[case_key]
        for channel in reversed(CHANNELS):
            record = pick(frame, metric, channel)
            rows.append((y, record, channel))
            ticks.append(y)
            labels.append(CHANNEL_LABEL[channel])
            y += 1.0
        # A gap and a case label between the two blocks.
        ax.text(0.02, y - 0.55, case_name, transform=ax.get_yaxis_transform(),
                fontsize=8.4, color=MUTED, va="center", ha="left")
        y += 1.0

    ax.axvline(0.0, color=AXIS, lw=1.0, zorder=1)
    for y_pos, record, channel in rows:
        colour = CHANNEL_COLOUR[channel]
        ax.plot([record.ci_low, record.ci_high], [y_pos, y_pos],
                color=colour, lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot([record["mean"]], [y_pos], marker="o", ms=8.0, color=colour,
                markeredgecolor="#fcfcfb", markeredgewidth=1.4, zorder=3)
        ax.annotate(fmt(record["mean"]), (record.ci_high, y_pos),
                    textcoords="offset points", xytext=(7, 0), va="center",
                    fontsize=7.8, color=INK)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_ylim(-0.8, y - 0.4)
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_title(title, fontsize=10, pad=8, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3)
    _headroom(ax, 0.30)

    if show_total:
        parts = []
        for case_key, tag in (("case1", "Case 1"), ("case2", "Case 2")):
            total = pick(data[case_key], metric, "total")
            parts.append(f"{tag} total {fmt(total['mean'])}")
        ax.text(0.0, -0.185, "   ·   ".join(parts), transform=ax.transAxes,
                fontsize=7.8, color=MUTED, va="top")


def gate_panel(ax, data, metric, title, xlabel):
    """Interaction only, under the two restoration-gate settings.

    Colour still means "interaction"; the gate setting is carried by fill, so no
    hue is reused for a second variable.
    """
    ticks, labels = [], []
    y = 0.0
    for case_key, gate_key, case_name in (
        ("case2", "case2_gate", "Case 2"),
        ("case1", "case1_gate", "Case 1"),
    ):
        for source, gate_name, filled in ((gate_key, "reads integrity", False),
                                          (case_key, "reads service", True)):
            record = pick(data[source], metric, "delta_int")
            colour = CHANNEL_COLOUR["delta_int"]
            ax.plot([record.ci_low, record.ci_high], [y, y], color=colour, lw=2.0,
                    solid_capstyle="round", zorder=2)
            ax.plot([record["mean"]], [y], marker="o", ms=8.0,
                    color=colour if filled else "#fcfcfb",
                    markeredgecolor=colour, markeredgewidth=1.8, zorder=3)
            ax.annotate(fmt(record["mean"]), (max(record.ci_high, record["mean"]), y),
                        textcoords="offset points", xytext=(7, 0), va="center",
                        fontsize=7.8, color=INK)
            ticks.append(y)
            labels.append(gate_name)
            y += 1.0
        ax.text(0.02, y - 0.55, case_name, transform=ax.get_yaxis_transform(),
                fontsize=8.4, color=MUTED, va="center", ha="left")
        y += 1.0

    ax.axvline(0.0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_ylim(-0.8, y - 0.4)
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_title(title, fontsize=10, pad=8, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3)
    _headroom(ax, 0.34)


def _headroom(ax, fraction: float) -> None:
    """Widen the x-range so annotations placed past the CI end stay inside."""
    low, high = ax.get_xlim()
    ax.set_xlim(low, high + fraction * (high - low))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="coupling0729/results/hpc_n1000")
    parser.add_argument("--out", default="coupling0729/results/hpc_n1000/fig5_attribution.png")
    args = parser.parse_args()

    data = load(Path(args.indir))

    # Three panels over two. Every panel here decomposes exactly; `T_rec` does
    # not, and is deliberately absent — see the module docstring.
    fig = plt.figure(figsize=(15.2, 8.2))
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2, 6, hspace=0.46, wspace=0.85,
                          left=0.072, right=0.975, top=0.875, bottom=0.115)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    forest_panel(axes[0], data, "loss",
                 "(a)  overall resilience loss  1 − R",
                 "effect on 1 − R  (loss direction)")
    forest_panel(axes[1], data, "Lambda",
                 "(b)  depth of drop  Λ",
                 "effect on Λ")
    forest_panel(axes[2], data, "lock_in",
                 "(c)  share never recovering within H",
                 "effect on lock-in probability")
    gate_panel(axes[3], data, "loss",
               "(d)  interaction under each gate setting",
               "Δ_int on 1 − R")
    gate_panel(axes[4], data, "lock_in",
               "(e)  interaction under each gate setting",
               "Δ_int on lock-in probability")

    handles = [
        Line2D([], [], color=CHANNEL_COLOUR[c], lw=2.0, marker="o", ms=8,
               markeredgecolor="#fcfcfb", markeredgewidth=1.4, label=CHANNEL_LABEL[c])
        for c in CHANNELS
    ] + [
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=8, markerfacecolor="#fcfcfb",
               markeredgecolor=MUTED, markeredgewidth=1.8,
               label="hollow = gate reads physical integrity"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.008))

    fig.suptitle(
        "Attribution of coupling-induced resilience loss, 1000 hazard replications per case\n"
        "bars are 95% bootstrap confidence intervals; every panel decomposes exactly. Recovery delay does not, "
        "and is shown as an empirical cumulative recovery curve in Figures 3–4",
        fontsize=11.5, y=0.985)
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
