"""Export the realised coupling edges and the environment that produced the runs.

The run records carry the *requested* design point, not the edge list that was
actually built from it, and not the library versions the physics came from.
Neither is recoverable after the fact from a CSV of metrics: breadth is quantised
by the target-layer size, edge placement is drawn from ``seed_design``, and the
power-flow and hydraulic solvers differ enough between versions to matter.

This rebuilds the operators from the same seeds and writes them out. It runs no
simulation, so it must be executed in the environment that produced the results
-- running it anywhere else would document the wrong environment.

    python3 examples/export_provenance.py --outdir results/provenance
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.core.scenario import CouplingDesign

# The design points that appear in the paper: the headline point, plus the grid.
Q_LEVELS = (0.20, 0.35, 0.50, 0.75, 1.00)
KAPPA_LEVELS = (0.20, 0.40, 0.60, 0.80, 1.00)


def versions() -> dict:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "node": platform.node(),
    }
    for name in ("numpy", "pandas", "scipy", "networkx", "pandapower", "wntr",
                 "matplotlib"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:                     # noqa: BLE001 - reported, not raised
            out[name] = f"absent ({type(exc).__name__})"
    return out


def edges_for(engine, scenario, design) -> list[dict]:
    positions, _ = engine._component_inventory()
    op_prop, op_rec = engine._build_operators(replace(scenario, design=design), positions)
    rows = []
    for operator, phase in ((op_prop, "propagation"), (op_rec, "restoration")):
        for edge in operator.edges:
            rows.append({
                "phase": phase,
                "source": edge.source,
                "target": edge.target,
                "kappa": float(edge.kappa),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/provenance")
    parser.add_argument("--grid", action="store_true",
                        help="also export every cell of the design sweep")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "environment.json").write_text(
        json.dumps(versions(), indent=2, sort_keys=True), encoding="utf-8")
    print("environment:", json.dumps(versions(), sort_keys=True))

    frames, summary = [], []
    for case in ("1", "2"):
        if case == "1":
            from coupling0729.cases.power_transport import base_scenario, build_case
        else:
            from coupling0729.cases.water_transport import base_scenario, build_case
        engine, scenario = build_case(), base_scenario()

        points = [(0.50, 0.80)]
        if args.grid:
            points = sorted({(q, k) for q in Q_LEVELS for k in KAPPA_LEVELS} | set(points))

        for q, kappa in points:
            design = CouplingDesign(q_prop=q, kappa_prop=kappa, q_rec=q, kappa_rec=kappa)
            rows = edges_for(engine, scenario, design)
            for row in rows:
                row.update({"case": case, "q_requested": q, "kappa_requested": kappa})
            frames.extend(rows)

            for phase in ("propagation", "restoration"):
                phase_rows = [r for r in rows if r["phase"] == phase]
                targets = {r["target"] for r in phase_rows}
                layer = engine.layers[
                    engine.restoration_spec.target_layer if phase == "restoration"
                    else engine.propagation_spec.target_layer]
                universe = (len(list(layer.components())) if phase == "restoration"
                            else len(list(layer.service_nodes())))
                summary.append({
                    "case": case, "phase": phase,
                    "q_requested": q, "kappa_requested": kappa,
                    "n_target_universe": universe,
                    "n_targets_coupled": len(targets),
                    "q_realised": len(targets) / universe if universe else float("nan"),
                    "n_edges": len(phase_rows),
                })

    pd.DataFrame(frames).to_csv(outdir / "coupling_edges.csv", index=False)
    realised = pd.DataFrame(summary)
    realised.to_csv(outdir / "realised_breadth.csv", index=False)
    print(f"\nwrote {len(frames)} edges to {outdir/'coupling_edges.csv'}")
    print(realised[realised.q_requested == 0.50].to_string(index=False))


if __name__ == "__main__":
    main()
