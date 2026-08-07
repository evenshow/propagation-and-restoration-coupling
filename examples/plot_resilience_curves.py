"""Figures 3 and 4: four-arm resilience curves, one panel per layer.

The replication shown is **not** arbitrary. With 1000 replications available,
picking the first one invites the question of whether it is representative, so
the run plotted is the one whose *uncoupled* loss is closest to the median
uncoupled loss — a typical hazard realisation, chosen on a quantity that no
coupling arm can influence. The selection rule is printed in the caption and the
lock-in rate is quoted beside it, so the reader can see whether the arm-11
behaviour shown is the common case or the exception.

Colour is shared with the attribution figure: arm 10 carries the propagation
hue, arm 01 the restoration hue, so the same mapping is read across figures. The
uncoupled reference is neutral and "both" is near-black; neither competes with a
mechanism colour.

    python3 examples/plot_resilience_curves.py --case 1 --indir results/case1_n1000
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

from coupling0729.core.scenario import ARM_NAMES

ARM_STYLE = {
    "00": ("#898781", (0, (5, 2)), "arm 00   no coupling"),
    "10": ("#2a78d6", "-", "arm 10   propagation only"),
    "01": ("#eb6834", "-", "arm 01   restoration only"),
    "11": ("#0b0b0b", "-", "arm 11   both"),
}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
HAZARD_BAND, EMBARGO_BAND = "#f0a30a", "#4a3aa7"


def median_replication(runs: pd.DataFrame) -> tuple[int, float]:
    """Replication whose uncoupled loss is nearest the median uncoupled loss."""
    baseline = runs[runs["arm"] == "00"]
    target = float(baseline["loss"].median())
    idx = (baseline["loss"] - target).abs().idxmin()
    return int(baseline.loc[idx, "replication"]), target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("1", "2"), required=True)
    parser.add_argument("--indir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.case == "1":
        from coupling0729.cases.power_transport import (
            base_scenario, build_case, case_weights,
        )
        layer_key, layer_name = "E:", "power (E)"
        title = "Case 1 — IEEE33 feeder and Sioux Falls road network under pluvial flooding"
    else:
        from coupling0729.cases.water_transport import (
            base_scenario, build_case, case_weights,
        )
        layer_key, layer_name = "W:", "water (W)"
        title = "Case 2 — EPANET Net3 and Sioux Falls road network under the same flood"

    indir = Path(args.indir)
    runs = pd.read_csv(indir / "runs.csv", dtype={"arm": str})
    runs["arm"] = runs["arm"].str.zfill(2)
    replication, median_loss = median_replication(runs)
    lock_in = float(runs[runs["arm"] == "11"]["lock_in"].mean())
    n_reps = int(runs["replication"].nunique())
    print(f"case {args.case}: median-representative replication = {replication} "
          f"(uncoupled loss {median_loss:.4f}); arm-11 lock-in rate {lock_in:.3f}")

    engine = build_case()
    weights = case_weights(engine)
    scenario = base_scenario()
    panels = {
        "system": weights,
        layer_name: {k: v for k, v in weights.items() if k.startswith(layer_key)},
        "transport (T)": {k: v for k, v in weights.items() if k.startswith("T:")},
    }

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    window = scenario.window

    for arm in ARM_NAMES:
        result = engine.run(scenario.for_replication(replication).for_arm(arm))
        colour, style, label = ARM_STYLE[arm]
        for ax, (name, w) in zip(axes, panels.items()):
            curve = result.system_curve(kind="operational", weights=w)
            ax.plot(curve.index, curve.to_numpy(), color=colour, ls=style, lw=2.0,
                    label=label, zorder=3 if arm == "11" else 2)

    # Fourth panel: recovery across *all* replications, not the single
    # realisation shown to the left. Differencing the conditional `T_rec`
    # differences subsamples of unequal size -- the interaction channel rests on
    # the runs in which arm 11 recovered at all, which are the milder floods, and
    # that selection can turn a real delay into an apparent speed-up. This curve
    # conditions on nothing: the rightward shift is the delay, and the plateau
    # below 100% is the lock-in.
    horizon = float(scenario.T_eval)
    for arm in ARM_NAMES:
        colour, style, label = ARM_STYLE[arm]
        values = runs.loc[runs["arm"] == arm, "T_rec"].to_numpy(dtype=float)
        recovered = np.sort(values[np.isfinite(values)])
        share = np.arange(1, recovered.size + 1) / values.size
        xs = np.concatenate(([0.0], recovered, [horizon]))
        ys = np.concatenate(([0.0], share, share[-1:] if share.size else [0.0]))
        axes[3].step(xs, ys, where="post", color=colour, ls=style, lw=2.0,
                     zorder=3 if arm == "11" else 2)
        # Label only the arms that fall short: the ones reaching the top say so
        # by touching it, and three "100%" labels would stack on one point.
        if ys[-1] < 0.99:
            axes[3].annotate(f"{ys[-1]:.0%}", (horizon, ys[-1]),
                             textcoords="offset points", xytext=(5, 0), va="center",
                             fontsize=8.6, color=colour, weight="bold")

    axes[3].set_xlim(0.0, horizon)
    axes[3].set_ylim(0.0, 1.04)
    axes[3].set_xlabel("T_rec   (minutes after t_ee)", fontsize=9)
    axes[3].set_ylabel(f"share of the {n_reps} replications recovered", fontsize=9.5)
    axes[3].set_title("recovery across all replications", fontsize=10.5, loc="left", pad=7)
    axes[3].grid(color=GRID, lw=0.7, zorder=0)
    axes[3].set_axisbelow(True)
    axes[3].set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        axes[3].spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes[3].spines[side].set_color(AXIS)
    axes[3].tick_params(colors=MUTED, labelcolor=INK, length=3, labelsize=8.5)

    view = window.t_oe + 0.62 * scenario.T_eval
    for ax, name in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.axvspan(window.t_oe, window.t_ee, color=HAZARD_BAND, alpha=0.16, lw=0, zorder=0)
        ax.axvspan(window.t_ee, window.t_repair_start, color=EMBARGO_BAND,
                   alpha=0.13, lw=0, zorder=0)
        ax.set_title(name, fontsize=10.5, loc="left", pad=7)
        ax.set_xlabel("t   (minutes)", fontsize=9)
        ax.set_xlim(window.t_oe - 40, view)
        ax.set_ylim(0.0, 1.04)
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=INK, length=3, labelsize=8.5)
        trans = ax.get_xaxis_transform()
        ax.text((window.t_oe + window.t_ee) / 2, 0.055, "I", transform=trans,
                ha="center", fontsize=8.5, color="#8a6d00", weight="bold")
        ax.text((window.t_ee + window.t_repair_start) / 2, 0.055, "II", transform=trans,
                ha="center", fontsize=8.5, color="#3f3190", weight="bold")
        ax.text(window.t_repair_start + 0.06 * (view - window.t_repair_start), 0.055,
                "III   restoration", transform=trans, ha="left", fontsize=8.5, color=MUTED)

    axes[0].set_ylabel("normalised performance  F(t)", fontsize=9.5)
    for ax in axes[1:3]:                       # same scale as axes[0], no relabelling
        ax.tick_params(labelleft=False)
    axes[0].legend(loc="lower right", fontsize=8.6, framealpha=0.95,
                   edgecolor=GRID, borderpad=0.6)

    fig.suptitle(
        f"{title}\n"
        f"panels 1–3: operational indicator, all four arms on one flood realisation "
        f"(the replication whose uncoupled loss is nearest the median of {n_reps})\n"
        f"panel 4: all {n_reps} replications, no conditioning — a rightward shift is "
        f"recovery delay, a plateau below 100% is lock-in "
        f"(arm 11 never recovers in {lock_in:.0%})",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
