"""Figure 2: the two cases on their shared geography, with both coupling operators.

Shows, per case: which components the flood damages directly (the intensity
field behind everything), where the two layers sit relative to each other, and
the direction of each dependency — propagation acting on service during the
event, restoration acting on repair afterwards.

The hazard field is drawn in neutral grey on purpose. Blue and orange are spoken
for: they carry the two mechanisms here and in the attribution figure, and a
blue hazard background would compete with the propagation arrows for the same
meaning.

Geography caveat, stated in the caption as well as here: Sioux Falls ships real
node coordinates, and Net3 has its own coordinate system which is mapped into
the road network's footprint. **IEEE33 has no coordinates at all** — its layout
is synthesised by embedding the electrical topology in the plane. It is a
plausible, reproducible feeder geography, not a real one.

    python3 examples/plot_case_geography.py --out results/fig2_geography.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.core.scenario import CouplingDesign

PROP_COLOUR = "#2a78d6"     # propagation, as in the attribution figure
REC_COLOUR = "#eb6834"      # restoration
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def mean_field(engine, xs, ys):
    """Deterministic mean of the hazard field; the residual is per-realisation."""
    field = engine.hazard_field
    gx, gy = np.meshgrid(xs, ys)
    centre = np.asarray(field.epicentre, dtype=float)
    dist = np.hypot(gx - centre[0], gy - centre[1])
    return field.peak * np.exp(-dist / max(field.attenuation, 1e-9))


def draw_case(ax, engine, design, source_layer, source_label, source_marker,
              service_layer_nodes, title):
    positions, classes = engine._component_inventory()
    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])
    pad = 0.6
    gx = np.linspace(xs.min() - pad, xs.max() + pad, 220)
    gy = np.linspace(ys.min() - pad, ys.max() + pad, 220)

    depth = mean_field(engine, gx, gy)
    mesh = ax.contourf(gx, gy, depth, levels=8, cmap="Greys", alpha=0.55, zorder=0)
    ax.contour(gx, gy, depth, levels=8, colors="#ffffff", linewidths=0.5, alpha=0.5, zorder=1)

    op_prop, op_rec = engine._build_operators(
        _scenario_with(engine, design), positions
    )

    # Layer marks. Damageable components are filled; service-only nodes hollow.
    def scatter(prefix, marker, size, label, facecolour, edgecolour):
        pts = [(p[0], p[1]) for name, p in positions.items() if name.startswith(prefix)]
        if not pts:
            return
        arr = np.asarray(pts)
        ax.scatter(arr[:, 0], arr[:, 1], marker=marker, s=size, zorder=4,
                   facecolor=facecolour, edgecolor=edgecolour, linewidths=0.8, label=label)

    scatter(source_layer, source_marker, 26, source_label, "#fcfcfb", INK)
    scatter("T:", "s", 30, "road node", "#fcfcfb", INK)

    for operator, colour, stem, alpha, width in (
        (op_prop, PROP_COLOUR, "D_prop  bus/pipe → road, acts on service", 0.75, 1.1),
        (op_rec, REC_COLOUR, "D_rec  road → bus/pipe, acts on repair", 0.32, 0.7),
    ):
        drawn = 0
        for edge in operator.edges:
            a, b = positions.get(edge.source), positions.get(edge.target)
            if a is None or b is None:
                continue
            ax.annotate("", xy=b, xytext=a, zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=width,
                                        alpha=alpha, shrinkA=3, shrinkB=3,
                                        connectionstyle="arc3,rad=0.12"))
            drawn += 1
        ax.plot([], [], color=colour, lw=1.8, alpha=max(alpha, 0.6),
                label=f"{stem}   ({drawn} edges)")

    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel("east   (km)", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3, labelsize=8.5)
    return mesh


def _scenario_with(engine, design):
    """A scenario carrying the design, for operator construction only."""
    from dataclasses import replace

    if "E" in engine.layers:
        from coupling0729.cases.power_transport import base_scenario
    else:
        from coupling0729.cases.water_transport import base_scenario
    return replace(base_scenario(), design=design)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/fig2_geography.png")
    args = parser.parse_args()

    from coupling0729.cases.power_transport import build_case as build_case1
    from coupling0729.cases.water_transport import build_case as build_case2

    design = CouplingDesign(q_prop=0.5, kappa_prop=0.8, q_rec=0.5, kappa_rec=0.8)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    draw_case(axes[0], build_case1(), design, "E:", "power bus", "o", None,
              "(a)  Case 1 — IEEE33 feeder  +  Sioux Falls")
    mesh = draw_case(axes[1], build_case2(), design, "W:pipe_", "water pipe", "D",
                     "W:junction_", "(b)  Case 2 — EPANET Net3  +  Sioux Falls")

    axes[0].set_ylabel("north   (km)", fontsize=9)
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
                  fontsize=8.2, frameon=False, handletextpad=0.6, columnspacing=1.6)

    bar = fig.colorbar(mesh, ax=axes, fraction=0.022, pad=0.02)
    bar.set_label("mean inundation depth  (m)", fontsize=8.8, color=INK)
    bar.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK)
    bar.outline.set_edgecolor(AXIS)

    fig.suptitle(
        "The two cases share one coordinate frame, so a spatially correlated flood damages both layers coherently\n"
        "D_prop acts on service during the event; D_rec acts on repair after it. "
        "Sioux Falls and Net3 carry real coordinates; the IEEE33 feeder layout is synthesised from its electrical topology.",
        fontsize=10.5, y=0.98)
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
