"""Case 1 (IEEE33 + Sioux Falls): the 2x2 factorial and the attribution table.

Block/merge execution, matching case 2: each block is the sequential path over a
disjoint set of replications, so it is restartable, individually inspectable and
maps one-to-one onto an SGE array task.

    for i in 0 1 2 3 4 5; do
      python coupling0729/examples/run_case1_factorial.py --replications 30 --block $i --n-blocks 6 &
    done
    python coupling0729/examples/run_case1_factorial.py --replications 30 --merge
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.power_transport import base_scenario, build_case, case_weights
from coupling0729.core.scenario import ARM_NAMES
from coupling0729.experiments import (
    attribution_audit,
    attribution_table,
    conditional_recovery_table,
    diagonal_dominance,
    effect_summary,
    paired_effects,
    superadditivity,
)
from coupling0729.metrics import result_metrics

METRICS = ("loss", "Lambda", "T_rec", "lock_in")

ARM_STYLE = {
    "00": ("#7f7f7f", "--", "arm 00  no coupling"),
    "10": ("#1f77b4", "-", "arm 10  propagation only"),
    "01": ("#d62728", "-", "arm 01  restoration only"),
    "11": ("#111111", "-", "arm 11  both"),
}


def plot_case1(engine, scenario, weights, out_path: Path) -> None:
    layers = {
        "system": weights,
        "power (E)": {k: v for k, v in weights.items() if k.startswith("E:")},
        "transport (T)": {k: v for k, v in weights.items() if k.startswith("T:")},
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharey=True)
    window = scenario.window
    for arm in ARM_NAMES:
        result = engine.run(scenario.for_replication(0).for_arm(arm))
        for ax, (title, w) in zip(axes, layers.items()):
            curve = result.system_curve(kind="operational", weights=w)
            colour, style, label = ARM_STYLE[arm]
            ax.plot(curve.index, curve.to_numpy(), color=colour, ls=style, lw=1.9, label=label)
            ax.set_title(title, fontsize=11)
    for ax in axes:
        ax.axvspan(window.t_oe, window.t_ee, color="#f0a30a", alpha=0.18, lw=0)
        ax.axvspan(window.t_ee, window.t_repair_start, color="#9467bd", alpha=0.15, lw=0)
        ax.set_xlabel("t  (minutes)")
        ax.set_xlim(window.t_oe - 40, window.t_oe + 0.6 * scenario.T_eval)
        ax.set_ylim(0.0, 1.04)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("normalised performance  F(t)")
    axes[0].legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    fig.suptitle(
        "Case 1 — IEEE33 feeder and Sioux Falls road network under pluvial flooding\n"
        "operational indicator, identical flood damage across arms (replication 0)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=155)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--flood-peak", type=float, default=0.60)
    parser.add_argument(
        "--gate-indicator", choices=("operational", "infrastructure"),
        default="operational",
        help="which state of the supporting layer the restoration gate reads",
    )
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--outdir", default="coupling0729/results/case1_factorial")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    blocks_dir = outdir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    engine = build_case(flood_peak=args.flood_peak, gate_indicator=args.gate_indicator)
    weights = case_weights(engine)
    scenario = base_scenario()

    if args.block is not None:
        indices = list(range(args.block, args.replications, args.n_blocks))
        print(f"block {args.block}/{args.n_blocks}: {len(indices)} replications", flush=True)
        started = time.time()
        rows = []
        for replication in indices:
            for arm in ARM_NAMES:
                result = engine.run(scenario.for_replication(replication).for_arm(arm))
                record = result_metrics(result, "operational", weights=weights)
                record["replication"] = replication
                record["frac_damaged"] = result.damage_summary.get("frac_damaged")
                rows.append(record)
            print(f"  replication {replication} done "
                  f"({time.time() - started:.0f}s elapsed)", flush=True)
        out = blocks_dir / f"runs_block_{args.block:03d}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out}", flush=True)
        return

    if args.merge:
        parts = sorted(blocks_dir.glob("runs_block_*.csv"))
        if not parts:
            raise SystemExit(f"no block CSVs in {blocks_dir}")
        # read_csv would infer an integer arm column and drop the leading zero,
        # turning "00" into 0 so the paired differencing loses its reference arm.
        runs = pd.concat(
            [pd.read_csv(f, dtype={"arm": str}) for f in parts], ignore_index=True
        )
        runs["arm"] = runs["arm"].astype(str).str.zfill(2)
        runs = runs.sort_values(["replication", "arm"]).reset_index(drop=True)
        expected = args.replications * len(ARM_NAMES)
        print(f"merged {len(parts)} blocks: {len(runs)} rows "
              f"({runs['replication'].nunique()} replications, expected {args.replications})",
              flush=True)
        if len(runs) != expected:
            print(f"  WARNING: expected {expected} rows, got {len(runs)} — "
                  "some array tasks may have failed", flush=True)
    else:
        started = time.time()
        rows = []
        for replication in range(args.replications):
            for arm in ARM_NAMES:
                result = engine.run(scenario.for_replication(replication).for_arm(arm))
                record = result_metrics(result, "operational", weights=weights)
                record["replication"] = replication
                rows.append(record)
            print(f"  replication {replication + 1}/{args.replications}", flush=True)
        runs = pd.DataFrame(rows)
        print(f"simulation took {time.time() - started:.0f}s", flush=True)

    effects = paired_effects(runs, metrics=METRICS)
    summary = effect_summary(effects, metrics=METRICS)
    table = attribution_table(effects, metrics=METRICS)
    audit = attribution_audit(effects, metrics=METRICS)
    delay = conditional_recovery_table(effects, metric="T_rec")
    h1 = diagonal_dominance(effects)
    h2 = superadditivity(effects, metrics=METRICS)

    runs.to_csv(outdir / "runs.csv", index=False)
    effects.to_csv(outdir / "effects.csv", index=False)
    summary.to_csv(outdir / "effect_summary.csv", index=False)
    table.to_csv(outdir / "attribution_table.csv")
    audit.to_csv(outdir / "attribution_audit.csv", index=False)
    delay.to_csv(outdir / "recovery_delay_conditional.csv", index=False)
    h1.to_csv(outdir / "h1_diagonal_dominance.csv", index=False)
    h2.to_csv(outdir / "h2_superadditivity.csv", index=False)
    try:
        plot_case1(engine, scenario, weights, outdir / "case1_arms.png")
    except Exception as exc:  # a failed figure must not lose the tables
        print(f"  [warn] figure skipped: {type(exc).__name__}: {exc}", flush=True)

    pd.set_option("display.width", 200)
    print("\n=== arm-level means ===")
    print(runs.groupby("arm")[["loss", "Lambda", "T_rec", "T_rec_censored", "F_end"]]
          .mean().round(4).to_string())
    print("\n=== attribution table ===")
    print(table.round(4).to_string())
    print("\n=== additivity audit ===")
    print(audit.round(6).to_string(index=False))
    print("\n=== recovery delay, conditional on recovering ===")
    print(delay.round(2).to_string(index=False))
    print("\n=== effect summary ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== H1 phase separation ===")
    print(h1.round(4).to_string(index=False))
    print("\n=== H2 super-additivity ===")
    print(h2.round(4).to_string(index=False))
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
