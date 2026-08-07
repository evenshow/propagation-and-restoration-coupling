"""Figure 2: layer topologies and the direction of each coupling operator.

Laid out as a layer diagram rather than a map. Each case is a row; the source
layer sits left, the transport layer right, and the coupling arrows cross the
whitespace between them. That keeps the arrows off the networks and lets the
figure do one job -- show what each layer looks like and which way each
dependency points.

Deliberately *not* shown: the hazard field, axes, gridlines, a colorbar. Putting
the flood field here would bury the topology under contours and invite the
reader to read the layout as a map. It is not one. Coordinates are a synthetic
spatial embedding: the road network carries its own coordinates, the IEEE33
feeder is laid out from its electrical topology (it ships none), and Net3's own
coordinates are rescaled into the road network's footprint. The two layers are
drawn apart for legibility; in the model they are co-located, which is what lets
one correlated flood hit both coherently.

Only three representative edges per operator are drawn. The simulations use the
full sets; drawing all of them turns each row into a hairball and hides the very
thing the figure exists to show.

    python3 examples/plot_case_topology.py --out results/fig2_topology.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.core.scenario import CouplingDesign

# Same mapping as Figures 1 and 3-5: blue is propagation, orange is restoration.
# Line style carries the same distinction, so the figure survives greyscale
# printing and does not rely on hue alone.
PROP, REC = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#898781"
NET_EDGE, NET_FACE, NET_RING = "#d8d7cf", "#f4f3ee", "#bcbab1"
SURFACE = "#fcfcfb"
N_SHOWN = 3


def unit_box(points):
    """Scale a layer into a unit box, preserving its shape (one scale factor)."""
    names = list(points)
    arr = np.array([points[n] for n in names], dtype=float)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = 1.0 / span.max()
    out = (arr - lo) * scale
    out[:, 0] += (1.0 - span[0] * scale) / 2
    out[:, 1] += (1.0 - span[1] * scale) / 2
    return {n: out[i] for i, n in enumerate(names)}


def affine_1d(source, target):
    """Recover the per-axis affine that maps one point set onto another."""
    a, b = np.polyfit(np.asarray(source, dtype=float), np.asarray(target, dtype=float), 1)
    return float(a), float(b)


def halo(size):
    return [pe.withStroke(linewidth=1.8, foreground=SURFACE)]


def draw_layer(ax, pos, links, title, node_size, labels=None):
    for a, b in links:
        if a in pos and b in pos:
            pa, pb = pos[a], pos[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=NET_EDGE, lw=1.0, zorder=1)
    arr = np.array([pos[n] for n in pos])
    ax.scatter(arr[:, 0], arr[:, 1], s=node_size, facecolor=NET_FACE,
               edgecolor=NET_RING, linewidths=0.7, zorder=2)
    if labels:
        for name in labels:
            if name in pos:
                p = pos[name]
                ax.text(p[0], p[1], labels[name], fontsize=5.2, ha="center", va="center",
                        color="#6d6b64", zorder=3, path_effects=halo(5.2))
    ax.text(0.5, -0.035, title, transform=ax.transAxes, fontsize=10.2, ha="center",
            va="top", color=INK, weight="bold")
    # Headroom at the top: the restoration arcs bulge upward, and without this
    # they ride over the row heading.
    x0, x1 = arr[:, 0].min(), arr[:, 0].max()
    y0, y1 = arr[:, 1].min(), arr[:, 1].max()
    span = max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - 0.06 * span, x1 + 0.06 * span)
    ax.set_ylim(y0 - 0.07 * span, y1 + 0.22 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor(SURFACE)


def mark_node(ax, pos, names, colour, marker="o", size=88):
    pts = [pos[n] for n in names if n in pos]
    if pts:
        arr = np.array(pts)
        ax.scatter(arr[:, 0], arr[:, 1], s=size, marker=marker, facecolor=colour,
                   edgecolor=SURFACE, linewidths=1.4, zorder=6)


def mark_pipe(ax, endpoints, colour):
    """Emphasise a selected pipe as a thickened segment, not as a point."""
    (x0, y0), (x1, y1) = endpoints
    ax.plot([x0, x1], [y0, y1], color=colour, lw=3.4, solid_capstyle="round", zorder=5)


def pick(edges, left_xy, right_xy, count):
    """Distinct at both ends, spread vertically, so arrows do not pile up.

    Deduplicating on the source alone is not enough: several road nodes can be
    nearest to the same pipe, which puts two arrowheads on one segment.
    """
    seen, unique = set(), []
    for edge in sorted(edges, key=lambda e: left_xy(e)[1] + right_xy(e)[1]):
        ends = (edge.source, edge.target)
        if ends[0] in seen or ends[1] in seen:
            continue
        seen.update(ends)
        unique.append(edge)
    if len(unique) <= count:
        return unique
    idx = np.linspace(0, len(unique) - 1, count).round().astype(int)
    return [unique[i] for i in dict.fromkeys(idx)]


def cross_arrow(fig, ax_a, xy_a, ax_b, xy_b, colour, dashed, rad):
    fig.add_artist(ConnectionPatch(
        xyA=xy_a, coordsA=ax_a.transData, axesA=ax_a,
        xyB=xy_b, coordsB=ax_b.transData, axesB=ax_b,
        arrowstyle="-|>", mutation_scale=15, linewidth=2.0, color=colour,
        linestyle="--" if dashed else "-", shrinkA=8, shrinkB=8, zorder=20,
        connectionstyle=f"arc3,rad={rad}"))


def build_case1():
    from coupling0729.cases.power_transport import base_scenario, build_case
    engine = build_case()
    power = engine.layers["E"]
    pos = unit_box(power.positions())
    links = [(str(r.from_bus), str(r.to_bus)) for r in power.sim.net.line.itertuples()]
    return dict(
        engine=engine, scenario=base_scenario(), prefix="E",
        pos=pos, links=links, node_size=32, labels={n: n for n in pos},
        title="Power layer  E  —  IEEE33 feeder", pipes=None,
    )


def build_case2():
    from coupling0729.cases.water_transport import base_scenario, build_case
    from coupling0729.cases.water_transport.layers import WaterLayer
    engine = build_case()
    water = engine.layers["W"]
    placed = water.positions()                       # shared frame
    raw = WaterLayer().positions()                   # Net3's own frame

    # positions() carries junctions and pipe midpoints only, so tanks and
    # reservoirs need the same affine applied by hand to close the pipe graph.
    shared = [n for n in raw if n in placed]
    ax_, bx = affine_1d([raw[n][0] for n in shared], [placed[n][0] for n in shared])
    ay_, by = affine_1d([raw[n][1] for n in shared], [placed[n][1] for n in shared])

    template = water._new_network()
    node_xy = {}
    for name in template.node_name_list:
        x, y = template.get_node(name).coordinates
        node_xy[f"node_{name}"] = (ax_ * float(x) + bx, ay_ * float(y) + by)

    combined = unit_box({**{k: tuple(v) for k, v in placed.items()}, **node_xy})
    pos = {k: v for k, v in combined.items() if not k.startswith("node_")}
    ends = {}
    for pipe in water.pipe_ids:
        link = template.get_link(pipe)
        a, b = f"node_{link.start_node_name}", f"node_{link.end_node_name}"
        if a in combined and b in combined:
            ends[f"pipe_{pipe}"] = (combined[a], combined[b])

    junctions = {k: v for k, v in pos.items() if k.startswith("junction_")}
    links = [(f"node_{template.get_link(p).start_node_name}",
              f"node_{template.get_link(p).end_node_name}") for p in water.pipe_ids]
    draw_pos = {k: v for k, v in combined.items() if k.startswith("node_")}
    return dict(
        engine=engine, scenario=base_scenario(), prefix="W",
        pos=pos, draw_pos=draw_pos, junctions=junctions, links=links,
        node_size=8, labels=None,
        title="Water layer  W  —  EPANET Net3", pipes=ends,
    )


def draw_row(fig, ax_left, ax_right, spec, design):
    engine = spec["engine"]
    traffic = engine.layers["T"]
    trf_pos = unit_box(traffic.positions())
    net = traffic.sim.net
    road_links = [(net.node_ids[a], net.node_ids[b]) for a, b in zip(net.fb, net.tb)]

    draw_layer(ax_left, spec.get("draw_pos", spec["pos"]), spec["links"], spec["title"],
               spec["node_size"], spec["labels"])
    if spec["pipes"] is not None:                    # junctions on top of the pipe graph
        arr = np.array(list(spec["junctions"].values()))
        ax_left.scatter(arr[:, 0], arr[:, 1], s=9, facecolor=NET_FACE,
                        edgecolor=NET_RING, linewidths=0.5, zorder=2)
    draw_layer(ax_right, trf_pos, road_links, "Transport layer  T  —  Sioux Falls",
               32, {n: n for n in trf_pos})

    positions, _ = engine._component_inventory()
    op_prop, op_rec = engine._build_operators(
        replace(spec["scenario"], design=design), positions)

    def bare(name):
        return name.split(":", 1)[1]

    def src_xy(name):
        key = bare(name)
        if spec["pipes"] and key in spec["pipes"]:
            (x0, y0), (x1, y1) = spec["pipes"][key]
            return np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        return spec["pos"][key]

    prop_edges = pick(op_prop.edges, lambda e: src_xy(e.source),
                      lambda e: trf_pos[bare(e.target)], N_SHOWN)
    rec_edges = pick(op_rec.edges, lambda e: src_xy(e.target),
                     lambda e: trf_pos[bare(e.source)], N_SHOWN)

    for edges, colour, source_side in ((prop_edges, PROP, "L"), (rec_edges, REC, "R")):
        on_left = [e.source for e in edges] if source_side == "L" else [e.target for e in edges]
        on_right = [e.target for e in edges] if source_side == "L" else [e.source for e in edges]
        for name in on_left:
            key = bare(name)
            if spec["pipes"] and key in spec["pipes"]:
                mark_pipe(ax_left, spec["pipes"][key], colour)
            else:
                mark_node(ax_left, spec["pos"], [key], colour)
        mark_node(ax_right, trf_pos, [bare(n) for n in on_right], colour, marker="s")

    for edge in prop_edges:                          # source layer -> transport
        cross_arrow(fig, ax_left, src_xy(edge.source), ax_right,
                    trf_pos[bare(edge.target)], PROP, dashed=False, rad=0.07)
    for edge in rec_edges:                           # transport -> source layer
        cross_arrow(fig, ax_right, trf_pos[bare(edge.source)], ax_left,
                    src_xy(edge.target), REC, dashed=True, rad=0.07)

    return len(op_prop.edges), len(op_rec.edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/fig2_topology.png")
    args = parser.parse_args()

    design = CouplingDesign(q_prop=0.5, kappa_prop=0.8, q_rec=0.5, kappa_rec=0.8)

    fig = plt.figure(figsize=(12.4, 11.2))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, wspace=0.14, hspace=0.24,
                          left=0.030, right=0.970, top=0.900, bottom=0.130)
    axes = [[fig.add_subplot(gs[r, c]) for c in (0, 1)] for r in (0, 1)]

    counts = []
    for row, spec, tag, tag_y in (
        (0, build_case1(), "(a)   Case 1  —  power and transport", 0.912),
        (1, build_case2(), "(b)   Case 2  —  water and transport", 0.505),
    ):
        counts.append(draw_row(fig, axes[row][0], axes[row][1], spec, design))
        fig.text(0.045, tag_y, tag, fontsize=11.5, ha="left", va="bottom",
                 color=INK, weight="bold")

    handles = [
        Line2D([], [], color=PROP, lw=2.0, ls="-",
               label="D_prop    source layer → transport,  acts on service during the event"),
        Line2D([], [], color=PROP, lw=0, marker="o", ms=8, markeredgecolor=SURFACE,
               label="propagation interface"),
        Line2D([], [], color=REC, lw=2.0, ls="--",
               label="D_rec    transport → source layer,  acts on repair afterwards"),
        Line2D([], [], color=REC, lw=0, marker="o", ms=8, markeredgecolor=SURFACE,
               label="restoration interface"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.088),
               ncol=2, frameon=False, fontsize=9.3, columnspacing=2.4,
               handletextpad=0.8, labelspacing=0.5)

    fig.suptitle("Layer topologies and the direction of each coupling operator",
                 fontsize=13.5, y=0.962)
    fig.text(0.5, 0.036,
             f"Only representative coupling edges are shown for clarity — three per operator; the "
             f"simulations use the full sets (case 1: {counts[0][0]} propagation and {counts[0][1]} "
             f"restoration edges; case 2: {counts[1][0]} and {counts[1][1]}).\n"
             "Coordinates are a synthetic spatial embedding in one common frame, not real geography. "
             "The layers are drawn apart for legibility; in the model they are co-located.",
             fontsize=8.4, color=MUTED, ha="center", va="top", linespacing=1.7)

    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
