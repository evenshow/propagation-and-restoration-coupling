"""Two targeted experiments on the gate, both testing predictions rather than
exploring.

**Where the transition sits.** Under the ``max`` semantics of Equation (3) a
single fully failed source gives ``s_j = 1 - kappa``, so the gate, which closes
at ``s_j < a_thr``, cannot close at all unless

    kappa > 1 - a_thr .

The transition observed in the design sweep should therefore sit at
``kappa_crit = 1 - a_thr`` and should *move* when the threshold moves. Running
three thresholds against one kappa grid tests that directly: a prediction that
survives is much stronger evidence than a transition found by looking.

**Whether lock-in is absorbing.** Lock-in is defined as failure to recover within
the horizon, which on its own cannot distinguish a latched gate from recovery
that is merely slow. Above ``kappa_crit`` the gate can hold shut permanently, so
extending the horizon should not rescue locked-in runs; below it, where the gate
cannot close and restoration coupling only derates the repair rate, a longer
horizon should. The ``--t-eval`` mode runs the same design on a much longer
horizon so the two can be told apart.

    python3 examples/run_gate_threshold.py --mode grid --task 0 --n-tasks 18
    python3 examples/run_gate_threshold.py --mode horizon --task 0 --n-tasks 10
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupling0729.cases.power_transport import base_scenario, build_case, case_weights
from coupling0729.core.scenario import ARM_NAMES, CouplingDesign
from coupling0729.metrics.headline import result_metrics

THRESHOLDS = (0.30, 0.40, 0.50)
KAPPAS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80)
BASELINE_Q = 0.50


def grid_points() -> list[tuple[float, float]]:
    return [(a, k) for a in THRESHOLDS for k in KAPPAS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("grid", "horizon"), required=True)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--t-eval", type=float, default=None,
                        help="override the evaluation horizon (horizon mode)")
    parser.add_argument("--access-threshold", type=float, default=0.40,
                        help="gate threshold in horizon mode")
    parser.add_argument("--kappa", type=float, default=0.80,
                        help="coupling intensity in horizon mode")
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--n-tasks", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir or f"results/case1_gate_{args.mode}")
    blocks = outdir / "blocks"
    blocks.mkdir(parents=True, exist_ok=True)

    if args.merge:
        frames = [pd.read_csv(f, dtype={"arm": str}) for f in sorted(blocks.glob("*.csv"))]
        if not frames:
            raise SystemExit(f"no block files under {blocks}")
        runs = pd.concat(frames, ignore_index=True)
        runs["arm"] = runs["arm"].str.zfill(2)
        runs.to_csv(outdir / "runs.csv", index=False)
        keys = ["access_threshold", "kappa"] if args.mode == "grid" else ["T_eval"]
        summary = (runs[runs.arm == "11"].groupby(keys)["lock_in"].agg(["mean", "size"])
                   .rename(columns={"mean": "lock_in_arm11", "size": "n"}))
        summary.to_csv(outdir / "summary.csv")
        print(f"merged {len(runs)} rows")
        print(summary.round(4).to_string())
        if args.mode == "grid":
            print("\npredicted kappa_crit = 1 - access_threshold:")
            for a in THRESHOLDS:
                print(f"   a_thr={a:.2f}  ->  kappa_crit={1 - a:.2f}")
        return

    if args.task is None:
        raise SystemExit("--task is required unless --merge is given")

    if args.mode == "grid":
        points = grid_points()
        per_point = max(1, args.n_tasks // len(points))
        index, block = divmod(args.task, per_point)
        if index >= len(points):
            print(f"task {args.task} maps past the grid; nothing to do")
            return
        a_thr, kappa = points[index]
        t_eval = None
    else:
        per_point, block = args.n_tasks, args.task
        a_thr, kappa, t_eval = args.access_threshold, args.kappa, args.t_eval

    engine = build_case(access_threshold=a_thr)
    weights = case_weights(engine)
    scenario = base_scenario()
    if t_eval is not None:
        scenario = replace(scenario, T_eval=float(t_eval), t_end=float(t_eval) + 400.0)
    scenario = replace(scenario, design=CouplingDesign(
        q_prop=BASELINE_Q, kappa_prop=kappa, q_rec=BASELINE_Q, kappa_rec=kappa))

    indices = list(range(block, args.replications, per_point))
    print(f"a_thr={a_thr} kappa={kappa} T_eval={scenario.T_eval} "
          f"block {block}/{per_point}: {len(indices)} replications", flush=True)

    rows, started = [], time.time()
    for replication in indices:
        per_rep = scenario.for_replication(replication)
        for arm in ARM_NAMES:
            result = engine.run(per_rep.for_arm(arm))
            record = result_metrics(result, "operational", weights=weights)
            record.update({"replication": replication, "access_threshold": a_thr,
                           "kappa": kappa, "T_eval": scenario.T_eval})
            rows.append(record)
        print(f"  replication {replication} done ({time.time() - started:.0f}s)", flush=True)

    out = blocks / f"a{int(a_thr * 100):03d}_k{int(kappa * 100):03d}_b{block:03d}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
