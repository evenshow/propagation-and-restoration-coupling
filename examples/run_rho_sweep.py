"""Where in the design space does the interaction dominate?

The headline attribution is measured at one design point. This sweep asks
whether interaction dominance is a property of that point or of the system, by
running the same 2x2 factorial across a ``q x kappa`` grid and reducing each
point to ``rho_int = E[Delta_int] / E[Delta_total]``.

Deliberately *not* built on ``run_case1_sweep.py``. That script exists to fit a
per-case absorption threshold ``kappa_0`` and test the ``psi`` collapse -- the
material cut from the paper as scope drift -- and its grid does not contain the
baseline design point, so neither its machinery nor its levels are reusable
here. What is reused is the differencing in ``paired_effects``, which carries
the ``min_count=3`` guard that keeps a censored arm from silently reading as a
zero effect.

The grid contains ``q = 0.5, kappa = 0.8`` so that one of its cells reproduces
the headline number and the sweep can be checked against a result computed by a
different runner at a different sample size.

    qsub -N rho1 -t 1-200 -v SGE_ARRAY_N=200 hpc/rho_array.sh --case 1
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.core.scenario import ARM_NAMES, CouplingDesign
from coupling0729.experiments import paired_effects, share_surface
from coupling0729.metrics.headline import result_metrics

Q_LEVELS = (0.20, 0.35, 0.50, 0.75, 1.00)
KAPPA_LEVELS = (0.20, 0.40, 0.60, 0.80, 1.00)
BASELINE = (0.50, 0.80)


def grid() -> list[tuple[float, float]]:
    points = [(q, k) for q in Q_LEVELS for k in KAPPA_LEVELS]
    if BASELINE not in points:                      # the cross-check would be lost
        raise AssertionError(f"grid must contain the baseline design point {BASELINE}")
    return points


def load_case(case: str):
    if case == "1":
        from coupling0729.cases.power_transport import base_scenario, build_case, case_weights
    else:
        from coupling0729.cases.water_transport import base_scenario, build_case, case_weights
    engine = build_case()
    return engine, case_weights(engine), base_scenario()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("1", "2"), required=True)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--task", type=int, default=None,
                        help="0-based flat index over (design point x replication block)")
    parser.add_argument("--n-tasks", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir or f"results/case{args.case}_rho_sweep")
    blocks = outdir / "blocks"
    blocks.mkdir(parents=True, exist_ok=True)
    points = grid()

    if args.merge:
        frames = [pd.read_csv(p, dtype={"arm": str}) for p in sorted(blocks.glob("*.csv"))]
        if not frames:
            raise SystemExit(f"no block files under {blocks}")
        runs = pd.concat(frames, ignore_index=True)
        runs["arm"] = runs["arm"].str.zfill(2)
        runs.to_csv(outdir / "runs.csv", index=False)

        design_cols = ["q_prop", "kappa_prop", "q_rec", "kappa_rec"]
        effects = []
        for values, group in runs.groupby(design_cols, sort=True):
            # paired_effects already returns `replication` as a column.
            frame = paired_effects(group, metrics=("loss", "Lambda", "lock_in"))
            for col, value in zip(design_cols, values):
                frame[col] = value
            effects.append(frame)
        effects = pd.concat(effects, ignore_index=True)
        effects.to_csv(outdir / "effects.csv", index=False)

        surface = share_surface(effects, metric="loss")
        surface.to_csv(outdir / "rho_int_surface.csv", index=False)

        expected = args.replications * len(points)
        got = int(runs.groupby(design_cols).ngroups)
        print(f"merged {len(runs)} rows over {got} design points "
              f"({len(runs) // 4} replications total, expected {expected})")
        cols = ["q_prop", "kappa_prop", "share", "share_ci_low", "share_ci_high",
                "mean_int", "mean_total", "n"]
        print(surface[cols].round(4).to_string(index=False))
        base = surface[(surface.q_prop == BASELINE[0]) & (surface.kappa_prop == BASELINE[1])]
        if not base.empty:
            row = base.iloc[0]
            print(f"\nbaseline cell q={BASELINE[0]} kappa={BASELINE[1]}: "
                  f"rho_int = {row['share']:.3f} "
                  f"[{row['share_ci_low']:.3f}, {row['share_ci_high']:.3f}]  "
                  f"(headline run gives 0.770 for case 1, 0.608 for case 2)")
        return

    # One task handles one design point and one slice of the replications, so a
    # task is short enough to fit a modest wall-clock request and a failure
    # costs one slice rather than a whole design point.
    n_blocks_per_point = max(1, args.n_tasks // len(points))
    if args.task is None:
        raise SystemExit("--task is required unless --merge is given")
    point_index, block = divmod(args.task, n_blocks_per_point)
    if point_index >= len(points):
        print(f"task {args.task} maps past the grid ({len(points)} points); nothing to do")
        return

    q, kappa = points[point_index]
    indices = list(range(block, args.replications, n_blocks_per_point))
    print(f"case {args.case}  q={q} kappa={kappa}  block {block}/{n_blocks_per_point}  "
          f"{len(indices)} replications", flush=True)

    engine, weights, scenario = load_case(args.case)
    design = CouplingDesign(q_prop=q, kappa_prop=kappa, q_rec=q, kappa_rec=kappa)
    scenario = replace(scenario, design=design)

    rows, started = [], time.time()
    for replication in indices:
        per_rep = scenario.for_replication(replication)
        for arm in ARM_NAMES:
            result = engine.run(per_rep.for_arm(arm))
            record = result_metrics(result, "operational", weights=weights)
            record.update({"replication": replication, "q_prop": q, "kappa_prop": kappa,
                           "q_rec": q, "kappa_rec": kappa})
            rows.append(record)
        print(f"  replication {replication} done ({time.time() - started:.0f}s)", flush=True)

    out = blocks / f"p{point_index:02d}_b{block:03d}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
