"""Four experiments answering the questions a reviewer is most likely to ask.

``layouts``
    The headline attribution rests on one realised set of coupling edges per
    case, drawn once from ``seed_design``, so the reported intervals cover flood
    realisations and nothing else. This repeats the baseline design point over
    several independent layouts, which is the only way to put a spread on
    ``rho_int`` across edge placements.

``cross``
    The 5 x 5 sweep gives both phases the same ``(q, kappa)``, so it moves along
    one diagonal of a four-dimensional design space and cannot say whether the
    transition is driven by the propagation intensity, the restoration intensity,
    or the pair. Crossing ``kappa_prop`` against ``kappa_rec`` at fixed breadth
    separates them; the off-diagonal cells are the informative ones, since a
    conjunction condition predicts they stay near zero.

``horizon``
    Case 1's lock-in was shown to be absorbing by extending the window fivefold.
    Case 2 was never tested the same way, so the claim currently generalises from
    one case. This runs the same test for case 2.

``multi_T``
    ``rho_int`` grows with the evaluation horizon because a locked-in run keeps
    accumulating loss. Recording ``1 - R`` at a ladder of horizons within one
    simulation makes that dependence a curve rather than an assertion, at no
    extra simulation cost.

    python3 examples/run_robustness.py --mode layouts --case 1 --task 0 --n-tasks 20
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.core.scenario import ARM_NAMES, CouplingDesign, SeedBundle
from coupling0729.metrics.headline import result_metrics

BASELINE_Q, BASELINE_K = 0.50, 0.80
LAYOUT_SEEDS = tuple(range(20))
CROSS_KAPPAS = (0.40, 0.60, 0.80, 1.00)
# Fractions of the simulated horizon, so the ladder is meaningful whatever
# `--t-eval` is set to; beyond the simulated end the curve has no more data and
# the integral would silently repeat its last value.
HORIZON_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def load_case(case: str):
    if case == "1":
        from coupling0729.cases.power_transport import base_scenario, build_case, case_weights
    else:
        from coupling0729.cases.water_transport import base_scenario, build_case, case_weights
    engine = build_case()
    return engine, case_weights(engine), base_scenario()


def loss_at_horizons(result, weights, ladder):
    """1 - R evaluated over a ladder of horizons from one trajectory.

    R is a time-average from ``t_oe`` to ``t_oe + T``; truncating the same curve
    at different T needs no re-simulation, only a different upper limit.
    """
    curve = result.system_curve(kind="operational", weights=weights)
    t = curve.index.to_numpy(dtype=float)
    f = curve.to_numpy(dtype=float)
    t0 = float(t[0])
    out = {}
    for T in ladder:
        mask = t <= t0 + float(T)
        if mask.sum() < 2:
            out[f"loss_T{T}"] = float("nan")
            continue
        out[f"loss_T{T}"] = float(1.0 - np.trapz(f[mask], t[mask]) / (t[mask][-1] - t0))
    return out


def points_for(mode, case):
    if mode == "layouts":
        return [{"design_seed": s} for s in LAYOUT_SEEDS]
    if mode == "cross":
        return [{"kappa_prop": a, "kappa_rec": b}
                for a in CROSS_KAPPAS for b in CROSS_KAPPAS]
    return [{}]                                   # horizon and multi_T: one cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("layouts", "cross", "horizon", "multi_T"),
                        required=True)
    parser.add_argument("--case", choices=("1", "2"), required=True)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--t-eval", type=float, default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--n-tasks", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir or f"results/case{args.case}_{args.mode}")
    blocks = outdir / "blocks"
    blocks.mkdir(parents=True, exist_ok=True)
    points = points_for(args.mode, args.case)

    if args.merge:
        frames = [pd.read_csv(f, dtype={"arm": str}) for f in sorted(blocks.glob("*.csv"))]
        if not frames:
            raise SystemExit(f"no block files under {blocks}")
        runs = pd.concat(frames, ignore_index=True)
        runs["arm"] = runs["arm"].str.zfill(2)
        runs.to_csv(outdir / "runs.csv", index=False)
        print(f"merged {len(runs)} rows, {len(runs) // 4} replications")
        return

    if args.task is None:
        raise SystemExit("--task is required unless --merge is given")

    per_point = max(1, args.n_tasks // len(points))
    index, block = divmod(args.task, per_point)
    if index >= len(points):
        print(f"task {args.task} maps past the design; nothing to do")
        return
    point = points[index]

    engine, weights, scenario = load_case(args.case)
    if args.t_eval is not None:
        scenario = replace(scenario, T_eval=float(args.t_eval),
                           t_end=float(args.t_eval) + 400.0)
    scenario = replace(scenario, design=CouplingDesign(
        q_prop=BASELINE_Q, q_rec=BASELINE_Q,
        kappa_prop=point.get("kappa_prop", BASELINE_K),
        kappa_rec=point.get("kappa_rec", BASELINE_K)))
    if "design_seed" in point:
        scenario = replace(scenario, seeds=SeedBundle(design=int(point["design_seed"])))

    indices = list(range(block, args.replications, per_point))
    print(f"{args.mode} case {args.case} {point} T={scenario.T_eval} "
          f"block {block}/{per_point}: {len(indices)} replications", flush=True)

    rows, started = [], time.time()
    for replication in indices:
        per_rep = scenario.for_replication(replication)
        for arm in ARM_NAMES:
            result = engine.run(per_rep.for_arm(arm))
            record = result_metrics(result, "operational", weights=weights)
            record.update({"replication": replication, "arm_check": arm, **point})
            if args.mode == "multi_T":
                ladder = [round(f * scenario.T_eval) for f in HORIZON_FRACTIONS]
                record.update(loss_at_horizons(result, weights, ladder))
            rows.append(record)
        print(f"  replication {replication} done ({time.time() - started:.0f}s)", flush=True)

    tag = "_".join(f"{k}{int(float(v) * 100):03d}" for k, v in sorted(point.items())) or "base"
    out = blocks / f"{tag}_b{block:03d}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
