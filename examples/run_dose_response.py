"""Dose-response sweeps: H3 (is psi = q*kappa sufficient?) and H5 (containment).

    .venv_coupling/Scripts/python.exe coupling0729/examples/run_dose_response.py --mode psi
    .venv_coupling/Scripts/python.exe coupling0729/examples/run_dose_response.py --mode threshold
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

from coupling0729.cases.toy2layer import base_scenario, build_case
from coupling0729.experiments import (
    collapse_diagnostic,
    containment_threshold,
    design_means,
    marginal_slopes,
    psi_grid,
    run_design_sweep,
    sweep_effects,
)

METRICS = ("loss", "Lambda", "T_rec", "lock_in")

PANELS = (
    ("Lambda__delta_prop", "prop", r"$\Delta_{prop}$ on $\Lambda$  (depth)"),
    ("T_rec__delta_rec", "rec", r"$\Delta_{rec}$ on $T_{rec}$  (delay)"),
)


# ---------------------------------------------------------------------------
def _surface(aggregated: pd.DataFrame, column: str, phase: str) -> pd.DataFrame:
    return aggregated.pivot_table(
        index=f"kappa_{phase}", columns=f"q_{phase}", values=column, aggfunc="mean"
    )


def plot_psi(
    aggregated: pd.DataFrame,
    diagnostics: pd.DataFrame,
    out_path: Path,
    per_seed: pd.DataFrame | None = None,
    placement: pd.DataFrame | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))

    for col, (column, phase, title) in enumerate(PANELS):
        grid = _surface(aggregated, column, phase)
        q_vals = grid.columns.to_numpy(float)
        k_vals = grid.index.to_numpy(float)

        ax = axes[0, col]
        mesh = ax.pcolormesh(q_vals, k_vals, grid.to_numpy(float), cmap="viridis", shading="nearest")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)

        # Iso-psi curves are hyperbolas kappa = psi / q. If psi is sufficient,
        # the colour contours must follow these lines.
        q_dense = np.linspace(max(q_vals.min(), 1e-3), q_vals.max(), 200)
        for psi in (0.1, 0.2, 0.35, 0.5, 0.7):
            kappa = psi / q_dense
            keep = (kappa >= k_vals.min()) & (kappa <= k_vals.max())
            if keep.sum() > 2:
                ax.plot(q_dense[keep], kappa[keep], color="w", lw=1.0, ls="--", alpha=0.75)
                j = np.argmax(keep)
                ax.text(q_dense[keep][0], kappa[keep][0], f" ψ={psi:g}",
                        color="w", fontsize=7.5, va="bottom")
        ax.set_xlabel(f"breadth  q ({phase})")
        ax.set_ylabel(f"intensity  κ ({phase})")
        ax.set_title(f"{title}\ndashed = iso-ψ hyperbolas", fontsize=10)

        ax = axes[1, col]
        sub = aggregated.dropna(subset=[column])

        # Individual layouts first, so the reader can see how much of the
        # within-iso-psi spread is merely placement noise.
        if per_seed is not None and column in per_seed.columns:
            layouts = per_seed.dropna(subset=[column])
            ax.scatter(layouts[f"psi_{phase}"], layouts[column], s=13,
                       color="0.55", alpha=0.55, linewidth=0, zorder=1,
                       label="individual layouts")
        scatter = ax.scatter(
            sub[f"psi_{phase}"], sub[column], c=sub[f"q_{phase}"],
            cmap="coolwarm", s=58, edgecolor="k", linewidth=0.5, zorder=3,
            label="composition mean",
        )
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label=f"q ({phase})")
        for psi_value, group in sub.groupby(sub[f"psi_{phase}"].round(9)):
            if len(group) >= 2:
                ax.plot([psi_value] * len(group), group[column],
                        color="0.35", lw=1.1, zorder=2)

        row = diagnostics[diagnostics["column"] == column]
        ratio = float(row["collapse_ratio"].iloc[0]) if len(row) else np.nan
        caption = f"naive ratio = {ratio:.3f}"
        if placement is not None:
            prow = placement[placement["column"] == column]
            if len(prow) and np.isfinite(prow["collapse_ratio_corrected"].iloc[0]):
                caption += (
                    f";  placement-corrected = {float(prow['collapse_ratio_corrected'].iloc[0]):.3f}"
                    f";  F = {float(prow['f_ratio'].iloc[0]):.2f}"
                )
        ax.set_xlabel(r"$\psi = q \cdot \kappa$")
        ax.set_ylabel(title)
        ax.set_title(f"collapse onto ψ\n{caption}", fontsize=9.5)
        ax.grid(alpha=0.25, lw=0.6)
        if per_seed is not None and col == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.suptitle(
        "H3: is ψ = q·κ a sufficient statistic for coupling? "
        "Top: effect surface over the (q, κ) plane.  Bottom: same points against ψ.",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out_path, dpi=155)
    plt.close(fig)


def plot_threshold(curves: pd.DataFrame, thresholds: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    cmap = plt.get_cmap("viridis")
    buffers = sorted(curves["buffer"].unique())

    ax = axes[0]
    for i, buffer in enumerate(buffers):
        group = curves[curves["buffer"] == buffer].sort_values("q_prop")
        ax.plot(group["q_prop"], group["Lambda__delta_prop"], marker="o", ms=5,
                color=cmap(i / max(len(buffers) - 1, 1)), label=f"buffer = {buffer:g}")
    ax.set_xlabel("propagation breadth  q")
    ax.set_ylabel(r"$\Delta_{prop}$ on $\Lambda$")
    ax.set_title("dose-response in q at each buffer level", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.6)

    ax = axes[1]
    ax.plot(thresholds["buffer"], thresholds["q_slope"], marker="s", color="#1f77b4")
    ax.axhline(0.0, color="0.5", lw=0.8)
    flattened = thresholds[thresholds["flattened"]]
    if len(flattened):
        first = float(flattened["buffer"].iloc[0])
        ax.axvline(first, color="#d62728", ls="--", lw=1.2)
        ax.annotate(f"containment threshold\nbuffer ≈ {first:g}",
                    xy=(first, float(flattened["q_slope"].iloc[0])),
                    xytext=(0.45, 0.72), textcoords="axes fraction", fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlabel("buffer capacity")
    ax.set_ylabel(r"$\partial \Delta_{prop} / \partial q$")
    ax.set_title("H5: does extra breadth stop mattering?", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=155)
    plt.close(fig)


# ---------------------------------------------------------------------------
def effective_coupling_table(composition: pd.DataFrame) -> pd.DataFrame:
    """Fit `psi_eff = q*(kappa - kappa_0)+` and cross-check kappa_0 two ways."""
    from coupling0729.experiments import (
        effective_coupling_fit,
        predicted_absorption_threshold,
    )

    rows = []
    for column, phase in (
        ("Lambda__delta_prop", "prop"),
        ("T_rec__delta_rec", "rec"),
        ("loss__delta_prop", "prop"),
        ("loss__delta_rec", "rec"),
    ):
        fit = effective_coupling_fit(composition, column, phase)
        fit.update(predicted_absorption_threshold(composition, column, phase))
        rows.append(fit)
    return pd.DataFrame(rows)


DIAGNOSTIC_TARGETS = (
    ("Lambda__delta_prop", "psi_prop"),
    ("T_rec__delta_rec", "psi_rec"),
    ("loss__delta_prop", "psi_prop"),
    ("loss__delta_rec", "psi_rec"),
    ("loss__delta_int", "psi_prop"),
)


def mode_psi(args, outdir: Path) -> None:
    from coupling0729.experiments import (
        COMPOSITION_COLUMNS,
        placement_decomposition,
        quantisation_report,
    )

    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = list(range(args.design_seeds)) if args.design_seeds > 1 else None
    points = psi_grid(levels, levels, design_seeds=seeds)
    engine = build_case()
    scenario = base_scenario()

    n_layouts = args.design_seeds if seeds else 1
    print(
        f"psi grid: {len(levels)**2} compositions x {n_layouts} layouts "
        f"= {len(points)} design points x {args.replications} replications x 4 arms"
    )
    sweep = run_design_sweep(
        engine, scenario, points, replications=args.replications, progress=True
    )
    effects = sweep_effects(sweep, metrics=METRICS)
    columns = [c for c in effects.columns if "__" in c]

    # Per-layout means, then composition means averaged over layouts.
    per_seed = design_means(effects, columns)
    composition = (
        per_seed.groupby(list(COMPOSITION_COLUMNS), as_index=False)[columns].mean()
        if seeds else per_seed.copy()
    )
    composition["psi_prop"] = composition["q_prop"] * composition["kappa_prop"]
    composition["psi_rec"] = composition["q_rec"] * composition["kappa_rec"]

    diagnostics = pd.DataFrame(
        [collapse_diagnostic(composition, col, psi) for col, psi in DIAGNOSTIC_TARGETS]
    )
    placement = (
        pd.DataFrame(
            [placement_decomposition(per_seed, col, psi) for col, psi in DIAGNOSTIC_TARGETS]
        )
        if seeds
        else pd.DataFrame()
    )
    restricted = pd.DataFrame(
        [
            {"kappa_floor": floor,
             **collapse_diagnostic(
                 composition[composition[f"kappa_{ph}"] >= floor - 1e-9], col, psi
             )}
            for floor in (0.2, 0.4, 0.6)
            for col, psi, ph in (
                ("Lambda__delta_prop", "psi_prop", "prop"),
                ("T_rec__delta_rec", "psi_rec", "rec"),
            )
        ]
    )
    slopes = pd.concat(
        [
            marginal_slopes(composition, "Lambda__delta_prop", "prop").assign(effect="Lambda_prop"),
            marginal_slopes(composition, "T_rec__delta_rec", "rec").assign(effect="T_rec_rec"),
        ],
        ignore_index=True,
    )
    effective = effective_coupling_table(composition)

    sweep.to_csv(outdir / "psi_runs.csv", index=False)
    effective.to_csv(outdir / "psi_effective_coupling.csv", index=False)
    effects.to_csv(outdir / "psi_effects.csv", index=False)
    per_seed.to_csv(outdir / "psi_per_layout_means.csv", index=False)
    composition.to_csv(outdir / "psi_design_means.csv", index=False)
    diagnostics.to_csv(outdir / "psi_collapse_diagnostics.csv", index=False)
    restricted.to_csv(outdir / "psi_collapse_by_kappa_floor.csv", index=False)
    slopes.to_csv(outdir / "psi_marginal_slopes.csv", index=False)
    if not placement.empty:
        placement.to_csv(outdir / "psi_placement_decomposition.csv", index=False)
    plot_psi(
        composition, diagnostics, outdir / "psi_collapse.png",
        per_seed=per_seed if seeds else None,
        placement=placement if not placement.empty else None,
    )

    pd.set_option("display.width", 200)
    print("\n=== H3 collapse diagnostics (composition means) ===")
    print(diagnostics.round(4).to_string(index=False))
    if not placement.empty:
        print("\n=== placement decomposition: is composition more than layout noise? ===")
        print(
            placement[
                ["column", "n_iso_psi_groups", "mean_layouts_per_composition",
                 "ms_within_placement", "ms_between_composition", "f_ratio", "p_value",
                 "collapse_ratio_corrected"]
            ].round(5).to_string(index=False)
        )
    print("\n=== collapse restricted by kappa floor ===")
    print(
        restricted[["kappa_floor", "column", "n_iso_psi_groups", "collapse_ratio",
                    "max_within_group_range"]].round(4).to_string(index=False)
    )
    print("\n=== effective coupling: psi_eff = q*(kappa - kappa_0)+ ===")
    print(
        effective[["column", "phase", "r2_psi", "kappa_0", "r2_psi_eff", "r2_gain",
                   "kappa_0_predicted"]].round(4).to_string(index=False)
    )
    print("\n=== marginal slopes ===")
    print(slopes.round(4).to_string(index=False))
    print("\n=== realised vs requested breadth ===")
    print(quantisation_report(sweep, "prop").round(6).to_string(index=False))
    print("\n=== censoring by arm ===")
    print(sweep.groupby("arm")["T_rec_censored"].mean().round(3).to_string())


def mode_threshold(args, outdir: Path) -> None:
    q_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    buffers = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    points = [
        {"q_prop": q, "kappa_prop": args.kappa, "q_rec": 0.0, "kappa_rec": 0.0}
        for q in q_values
    ]

    frames = []
    for buffer in buffers:
        print(f"buffer = {buffer:g}", flush=True)
        engine = build_case(buffer_capacity=buffer)
        # Only arms 00 and 10 are needed: Delta_prop is all this sweep reads,
        # and arm 10 has the restoration operator switched off entirely.
        sweep = run_design_sweep(
            engine, base_scenario(), points,
            replications=args.replications, arms=("00", "10"),
        )
        effects = sweep_effects(sweep, metrics=("loss", "Lambda"))
        aggregated = design_means(
            effects, [c for c in effects.columns if "__" in c]
        )
        aggregated["buffer"] = buffer
        frames.append(aggregated)

    curves = pd.concat(frames, ignore_index=True)
    thresholds = containment_threshold(curves, "buffer")

    curves.to_csv(outdir / "threshold_curves.csv", index=False)
    thresholds.to_csv(outdir / "threshold_slopes.csv", index=False)
    plot_threshold(curves, thresholds, outdir / "containment_threshold.png")

    print("\n=== H5 containment threshold ===")
    print(thresholds.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("psi", "threshold"), default="psi")
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument(
        "--design-seeds", type=int, default=1,
        help="number of edge layouts per (q, kappa) composition; >1 enables the "
             "placement decomposition that separates layout noise from psi-insufficiency",
    )
    parser.add_argument("--kappa", type=float, default=0.8)
    parser.add_argument("--outdir", type=str, default="coupling0729/results/dose_response")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (mode_psi if args.mode == "psi" else mode_threshold)(args, outdir)
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
