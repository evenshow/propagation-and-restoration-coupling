"""Validate the framework end to end on the two-layer toy case.

Produces:
  1. a representative four-arm resilience curve (does the trapezoid appear?);
  2. the 3x3 phase x mechanism attribution table;
  3. the H1 diagonal-dominance test and the H4 super-additivity test.

Run with the project virtual environment::

    .venv_coupling/Scripts/python.exe coupling0729/examples/run_toy_factorial.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.toy2layer import base_scenario, build_case
from coupling0729.core.scenario import ARM_NAMES
from coupling0729.experiments import (
    attribution_table,
    diagonal_dominance,
    effect_summary,
    paired_effects,
    run_factorial,
    superadditivity,
)

ARM_STYLE = {
    "00": ("#7f7f7f", "--", "arm 00  no coupling"),
    "10": ("#1f77b4", "-", "arm 10  propagation only"),
    "01": ("#d62728", "-", "arm 01  restoration only"),
    "11": ("#111111", "-", "arm 11  both"),
}


def plot_arms(trajectories, scenario, out_path: Path, t_view: float = 380.0) -> None:
    window = scenario.window
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)

    for ax, kind in zip(axes, ("operational", "infrastructure")):
        for arm in ARM_NAMES:
            curve = trajectories[(0, arm)].system_curve(kind=kind)
            colour, style, label = ARM_STYLE[arm]
            ax.plot(curve.index, curve.to_numpy(), color=colour, ls=style, lw=2.0, label=label)

        ax.axvspan(window.t_oe, window.t_ee, color="#f0a30a", alpha=0.18, lw=0)
        ax.axvspan(window.t_ee, window.t_repair_start, color="#9467bd", alpha=0.15, lw=0)
        ax.axvline(window.t_ee, color="#444", lw=0.9, ls=":")
        ax.set_title(f"{kind} indicator", fontsize=11)
        ax.set_xlabel("t")
        ax.set_xlim(window.t_oe - 8, t_view)
        ax.set_ylim(0.0, 1.04)
        ax.grid(alpha=0.25, lw=0.6)

        # Phase labels live in axes-fraction y so they never distort the layout.
        trans = ax.get_xaxis_transform()
        ax.text((window.t_oe + window.t_ee) / 2, 0.06, "I", transform=trans,
                ha="center", fontsize=9, color="#8a6d00", weight="bold")
        ax.text((window.t_ee + window.t_repair_start) / 2, 0.06, "II", transform=trans,
                ha="center", fontsize=9, color="#5b3a8a", weight="bold")
        ax.text(window.t_repair_start + 0.10 * (t_view - window.t_repair_start), 0.06,
                "III  restoration", transform=trans, ha="left", fontsize=9, color="#444")

    lock_level = float(trajectories[(0, "11")].system_curve(kind="operational").iloc[-1])
    if lock_level < 0.99:
        axes[0].annotate(
            f"arm 11 never recovers\n(locks in at F = {lock_level:.2f})",
            xy=(t_view * 0.86, lock_level), xytext=(t_view * 0.42, 0.35),
            fontsize=9, color="#111",
            arrowprops=dict(arrowstyle="->", color="#111", lw=1.0),
        )

    axes[0].set_ylabel("normalised performance  F(t)")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.suptitle(
        "Two-layer validation case: the four coupling arms under identical hazard damage "
        "(replication 0)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=60)
    parser.add_argument("--q-prop", type=float, default=0.6)
    parser.add_argument("--kappa-prop", type=float, default=0.8)
    parser.add_argument("--q-rec", type=float, default=0.6)
    parser.add_argument("--kappa-rec", type=float, default=0.8)
    parser.add_argument("--outdir", type=str, default="coupling0729/results/toy_factorial")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    engine = build_case()
    scenario = base_scenario(
        q_prop=args.q_prop,
        kappa_prop=args.kappa_prop,
        q_rec=args.q_rec,
        kappa_rec=args.kappa_rec,
    )

    print(f"running {args.replications} replications x 4 arms ...")
    runs, trajectories = run_factorial(
        engine, scenario, replications=args.replications, keep_trajectories=True
    )
    metrics = ("loss", "Lambda", "T_rec", "lock_in")
    effects = paired_effects(runs, metrics=metrics)
    summary = effect_summary(effects, metrics=metrics)
    table = attribution_table(effects, metrics=metrics)
    h1 = diagonal_dominance(effects)
    h4 = superadditivity(effects)

    runs.to_csv(outdir / "runs.csv", index=False)
    effects.to_csv(outdir / "effects.csv", index=False)
    summary.to_csv(outdir / "effect_summary.csv", index=False)
    table.to_csv(outdir / "attribution_table.csv")
    h1.to_csv(outdir / "h1_diagonal_dominance.csv", index=False)
    h4.to_csv(outdir / "h4_superadditivity.csv", index=False)
    plot_arms(trajectories, scenario, outdir / "arms_curve.png")

    pd.set_option("display.width", 160)
    print("\n=== arm-level means (loss direction) ===")
    print(
        runs.groupby("arm")[["loss", "Lambda", "T_rec", "T_rec_censored"]]
        .mean()
        .round(4)
        .to_string()
    )
    print("\n=== 3x3 attribution table (mean over replications) ===")
    print(table.round(4).to_string())
    print("\n=== effect summary with bootstrap CI ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== H1 phase separation (diagonal dominance) ===")
    print(h1.round(4).to_string(index=False))
    print("\n=== H4 super-additivity of the two couplings ===")
    print(h4.round(4).to_string(index=False))
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
