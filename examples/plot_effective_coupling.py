"""Figure: psi = q*kappa versus psi_eff = q*(kappa - kappa_0)+.

Reads the saved sweep outputs, so it is cheap to re-run and keeps analysis
separate from simulation.

    .venv_coupling/Scripts/python.exe coupling0729/examples/plot_effective_coupling.py \
        --indir coupling0729/results/dose_response_multiseed
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

from coupling0729.experiments import effective_coupling_fit, predicted_absorption_threshold

TARGETS = (
    ("Lambda__delta_prop", "prop", r"$\Delta_{prop}$ on $\Lambda$"),
    ("T_rec__delta_rec", "rec", r"$\Delta_{rec}$ on $T_{rec}$"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="coupling0729/results/dose_response_multiseed")
    args = parser.parse_args()
    indir = Path(args.indir)

    composition = pd.read_csv(indir / "psi_design_means.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))

    for row, (column, phase, label) in enumerate(TARGETS):
        fit = effective_coupling_fit(composition, column, phase)
        predicted = predicted_absorption_threshold(composition, column, phase)
        q = composition[f"q_{phase}"].to_numpy(float)
        kappa = composition[f"kappa_{phase}"].to_numpy(float)
        y = composition[column].to_numpy(float)

        for col, (x, name, r2) in enumerate(
            (
                (q * kappa, r"$\psi = q\,\kappa$", fit["r2_psi"]),
                (
                    q * np.clip(kappa - fit["kappa_0"], 0.0, None),
                    rf"$\psi_{{eff}} = q\,(\kappa - {fit['kappa_0']:.2f})^{{+}}$",
                    fit["r2_psi_eff"],
                ),
            )
        ):
            ax = axes[row, col]
            scatter = ax.scatter(x, y, c=kappa, cmap="viridis", s=55,
                                 edgecolor="k", linewidth=0.4, zorder=3)
            if col == 1:
                fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03, label="κ")

            slope = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0
            grid = np.linspace(0.0, x.max() * 1.03, 50)
            ax.plot(grid, slope * grid, color="#d62728", lw=1.4, zorder=2,
                    label=f"fit through origin, $R^2$ = {r2:.3f}")
            ax.set_xlabel(name)
            ax.set_ylabel(label)
            ax.legend(fontsize=8.5, loc="upper left")
            ax.grid(alpha=0.25, lw=0.6)

        axes[row, 1].set_title(
            f"κ₀ fitted = {fit['kappa_0']:.3f};  "
            f"κ₀ from marginal slopes = {predicted['kappa_0_predicted']:.3f}",
            fontsize=9.5,
        )
        axes[row, 0].set_title("breadth and intensity as substitutes", fontsize=9.5)

    fig.suptitle(
        "Breadth and intensity are substitutes only above an absorption threshold:\n"
        r"replacing $\psi = q\kappa$ by $\psi_{eff} = q(\kappa-\kappa_0)^+$ collapses the surface onto one line",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = indir / "effective_coupling.png"
    fig.savefig(out, dpi=155)
    plt.close(fig)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
