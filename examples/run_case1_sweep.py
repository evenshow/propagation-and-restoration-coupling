"""Case 1 dose-response: fit the feeder-and-roads case's own ``kappa_0``.

Reduced grid. The point is not to redraw the toy case's surface at high
resolution but to answer one question: does case 1 have an absorption threshold
of its own, and is it recovered consistently by two independent routes? That is
the precondition for the cross-case collapse test, which must be run against
``psi_eff = q(kappa - kappa_0)+`` with a per-case ``kappa_0``.

    .venv_coupling/Scripts/python.exe coupling0729/examples/run_case1_sweep.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.power_transport import base_scenario, build_case, case_weights
from coupling0729.experiments import (
    collapse_diagnostic,
    replication_rows,
    design_means,
    effective_coupling_fit,
    marginal_slopes,
    placement_decomposition,
    predicted_absorption_threshold,
    psi_grid,
    quantisation_report,
    sweep_effects,
)

METRICS = ("loss", "Lambda", "T_rec", "lock_in")

TARGETS = (
    ("Lambda__delta_prop", "prop", r"$\Delta_{prop}$ on $\Lambda$"),
    ("T_rec__delta_rec", "rec", r"$\Delta_{rec}$ on $T_{rec}$"),
)


def plot_case1_sweep(composition: pd.DataFrame, per_layout: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))
    for row, (column, phase, label) in enumerate(TARGETS):
        fit = effective_coupling_fit(composition, column, phase)
        predicted = predicted_absorption_threshold(composition, column, phase)
        q = composition[f"q_{phase}"].to_numpy(float)
        kappa = composition[f"kappa_{phase}"].to_numpy(float)
        y = composition[column].to_numpy(float)

        for col, (x, name, r2) in enumerate((
            (q * kappa, r"$\psi = q\,\kappa$", fit["r2_psi"]),
            (q * np.clip(kappa - fit["kappa_0"], 0.0, None),
             rf"$\psi_{{eff}} = q\,(\kappa - {fit['kappa_0']:.2f})^{{+}}$", fit["r2_psi_eff"]),
        )):
            ax = axes[row, col]
            if column in per_layout.columns:
                lay = per_layout.dropna(subset=[column])
                lx = (lay[f"q_{phase}"].to_numpy(float)
                      * (lay[f"kappa_{phase}"].to_numpy(float) if col == 0
                         else np.clip(lay[f"kappa_{phase}"].to_numpy(float) - fit["kappa_0"], 0, None)))
                ax.scatter(lx, lay[column], s=14, color="0.6", alpha=0.6, linewidth=0, zorder=1)
            sc = ax.scatter(x, y, c=kappa, cmap="viridis", s=58,
                            edgecolor="k", linewidth=0.4, zorder=3)
            if col == 1:
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, label="κ")
            denom = float((x * x).sum())
            slope = float((x * y).sum() / denom) if denom > 0 else 0.0
            grid = np.linspace(0.0, max(x.max(), 1e-9) * 1.03, 40)
            ax.plot(grid, slope * grid, color="#d62728", lw=1.4, zorder=2,
                    label=f"fit through origin, $R^2$ = {r2:.3f}")
            ax.set_xlabel(name)
            ax.set_ylabel(label)
            ax.legend(fontsize=8.5, loc="upper left")
            ax.grid(alpha=0.25, lw=0.6)
        axes[row, 1].set_title(
            f"κ₀ fitted = {fit['kappa_0']:.3f};  from marginal slopes = "
            f"{predicted['kappa_0_predicted']:.3f}", fontsize=9.5)
        axes[row, 0].set_title("breadth and intensity as substitutes", fontsize=9.5)

    fig.suptitle(
        "Case 1 (IEEE33 + Sioux Falls): does the real case have its own absorption threshold?",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=155)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=15)
    parser.add_argument("--design-seeds", type=int, default=2)
    parser.add_argument("--block", type=int, default=None,
                        help="run only replications with index %% n_blocks == block")
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="merge the per-block CSVs and run the analysis")
    parser.add_argument("--flood-peak", type=float, default=0.60)
    parser.add_argument("--outdir", default="coupling0729/results/case1_sweep")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    levels = [0.25, 0.60, 1.00]
    seeds = list(range(args.design_seeds)) if args.design_seeds > 1 else None
    points = psi_grid(levels, levels, design_seeds=seeds)
    factory_kwargs = {"flood_peak": args.flood_peak}
    weights = case_weights(build_case(**factory_kwargs))
    scenario = base_scenario()

    blocks_dir = outdir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    # Independent OS processes rather than a process pool: each block is exactly
    # the sequential path, so it is restartable, individually inspectable, and
    # maps one-to-one onto an HPC array task. A pool was tried first and its
    # workers were terminated abruptly on this machine for reasons that survived
    # ruling out memory, run count, specific configurations and BLAS
    # oversubscription; independent processes sidestep the question entirely.
    if args.block is not None:
        indices = list(range(args.block, args.replications, args.n_blocks))
        engine = build_case(**factory_kwargs)
        print(f"block {args.block}/{args.n_blocks}: replications {indices}", flush=True)
        started = time.time()
        rows = []
        for replication in indices:
            rows += replication_rows(
                engine, scenario, points, replication, weights=weights
            )
            print(f"  replication {replication} done "
                  f"({time.time() - started:.0f}s elapsed)", flush=True)
        out = blocks_dir / f"runs_block_{args.block:02d}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out}", flush=True)
        return

    if args.merge:
        parts = sorted(blocks_dir.glob("runs_block_*.csv"))
        if not parts:
            raise SystemExit(f"no block CSVs found in {blocks_dir}")
        # Arm names are "00"/"01"/"10"/"11". Left to itself read_csv infers an
        # integer column and silently drops the leading zero, so "00" comes back
        # as 0 and the paired differencing cannot find its reference arm.
        sweep = pd.concat(
            [pd.read_csv(f, dtype={"arm": str}) for f in parts], ignore_index=True
        )
        sweep["arm"] = sweep["arm"].astype(str).str.zfill(2)
        sort_keys = [c for c in ("replication", "q_prop", "kappa_prop", "seed_design", "arm")
                     if c in sweep.columns]
        sweep = sweep.sort_values(sort_keys).reset_index(drop=True)
        print(f"merged {len(parts)} blocks: {len(sweep)} rows, "
              f"{sweep['replication'].nunique()} replications", flush=True)
    else:
        n_runs = args.replications * (len(points) * 3 + 1)
        print(f"case 1 sweep: {len(levels)**2} compositions x {len(seeds or [0])} layouts "
              f"x {args.replications} replications = {n_runs} runs (sequential)", flush=True)
        engine = build_case(**factory_kwargs)
        started = time.time()
        rows = []
        for replication in range(args.replications):
            rows += replication_rows(engine, scenario, points, replication, weights=weights)
            print(f"  replication {replication + 1}/{args.replications}", flush=True)
        sweep = pd.DataFrame(rows)
        print(f"simulation took {time.time() - started:.0f}s", flush=True)

    effects = sweep_effects(sweep, metrics=METRICS)
    columns = [c for c in effects.columns if "__" in c]
    per_layout = design_means(effects, columns)
    composition = per_layout.groupby(
        ["q_prop", "kappa_prop", "q_rec", "kappa_rec"], as_index=False
    )[columns].mean()
    composition["psi_prop"] = composition["q_prop"] * composition["kappa_prop"]
    composition["psi_rec"] = composition["q_rec"] * composition["kappa_rec"]

    fits = pd.DataFrame([
        {**effective_coupling_fit(composition, col, ph),
         **predicted_absorption_threshold(composition, col, ph)}
        for col, ph in (("Lambda__delta_prop", "prop"), ("T_rec__delta_rec", "rec"),
                        ("loss__delta_prop", "prop"), ("loss__delta_rec", "rec"))
    ])
    diagnostics = pd.DataFrame([
        collapse_diagnostic(composition, "Lambda__delta_prop", "psi_prop"),
        collapse_diagnostic(composition, "T_rec__delta_rec", "psi_rec"),
    ])
    placement = pd.DataFrame([
        placement_decomposition(per_layout, "Lambda__delta_prop", "psi_prop"),
        placement_decomposition(per_layout, "T_rec__delta_rec", "psi_rec"),
    ]) if seeds else pd.DataFrame()
    slopes = marginal_slopes(composition, "Lambda__delta_prop", "prop")

    sweep.to_csv(outdir / "runs.csv", index=False)
    effects.to_csv(outdir / "effects.csv", index=False)
    per_layout.to_csv(outdir / "per_layout_means.csv", index=False)
    composition.to_csv(outdir / "design_means.csv", index=False)
    fits.to_csv(outdir / "effective_coupling.csv", index=False)
    diagnostics.to_csv(outdir / "collapse_diagnostics.csv", index=False)
    slopes.to_csv(outdir / "marginal_slopes.csv", index=False)
    if not placement.empty:
        placement.to_csv(outdir / "placement_decomposition.csv", index=False)
    plot_case1_sweep(composition, per_layout, outdir / "case1_effective_coupling.png")

    pd.set_option("display.width", 210)
    print("\n=== effect surface: Lambda__delta_prop (rows=kappa, cols=q) ===")
    print(composition.pivot_table(index="kappa_prop", columns="q_prop",
                                  values="Lambda__delta_prop").round(4).to_string())
    print("\n=== effective coupling: psi_eff = q*(kappa - kappa_0)+ ===")
    print(fits[["column", "phase", "r2_psi", "kappa_0", "r2_psi_eff", "r2_gain",
                "kappa_0_predicted"]].round(4).to_string(index=False))
    print("\n=== collapse onto psi ===")
    print(diagnostics.round(4).to_string(index=False))
    if not placement.empty:
        print("\n=== placement decomposition ===")
        print(placement[["column", "n_iso_psi_groups", "f_ratio", "p_value",
                         "collapse_ratio_corrected"]].round(4).to_string(index=False))
    print("\n=== marginal slopes ===")
    print(slopes.round(4).to_string(index=False))
    print("\n=== requested vs realised breadth ===")
    print(quantisation_report(sweep, "prop").round(4).to_string(index=False))
    print(quantisation_report(sweep, "rec").round(4).to_string(index=False))
    print("\n=== censoring by arm ===")
    print(sweep.groupby("arm")["T_rec_censored"].mean().round(3).to_string())
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
