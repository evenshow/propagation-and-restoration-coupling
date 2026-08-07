#!/bin/bash -l
#
# Does closing pipes change *junction* demand under wntr 1.0.0?
# Run as: bash -l $HOME/probe_wntr_close.sh
#
# Note: summing demand over ALL nodes is ~0 by mass conservation (reservoir and
# tank inflows cancel junction draw), so it says nothing. Only junction demand
# is informative, which is what the adapter actually uses.

set -u
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":${PYTHONPATH:-}

python3 - <<'PY'
import wntr
from wntr.network import LinkStatus

import os
INP = os.path.join(os.environ.get("PROJECT_ROOT",
                                  os.path.expanduser("~/coupling0729")),
                   "data", "net3", "Net3.inp")
print("wntr", wntr.__version__)


def build():
    wn = wntr.network.WaterNetworkModel(INP)
    wn.options.time.duration = 0
    wn.options.time.hydraulic_timestep = 3600
    wn.options.hydraulic.demand_model = "PDD"
    return wn


def junction_demand(wn):
    res = wntr.sim.WNTRSimulator(wn).run_sim()
    frame = res.node["demand"]
    if len(frame) == 0:
        return None, None
    row = frame.iloc[0]
    served = {j: float(row.get(j, 0.0)) for j in wn.junction_name_list}
    return sum(served.values()), served


base = build()
pipes = list(base.pipe_name_list)
total, baseline = junction_demand(base)
print("junctions=%d pipes=%d" % (len(base.junction_name_list), len(pipes)))
print("baseline junction demand total = %.6f" % total)
print("baseline junctions with demand > 1e-9: %d"
      % sum(1 for v in baseline.values() if v > 1e-9))
print()

for count in (12, 40, len(pipes)):
    wn = build()
    for name in pipes[:count]:
        wn.get_link(name).initial_status = LinkStatus.Closed
    total_c, served = junction_demand(wn)
    if total_c is None:
        print("closed %3d pipes -> hydraulics INFEASIBLE (empty result)" % count)
        continue
    dropped = sum(
        1 for j, v in served.items()
        if baseline.get(j, 0.0) > 1e-9 and v < baseline[j] - 1e-9
    )
    print("closed %3d pipes -> junction demand total = %.6f  (%.1f%% of baseline), "
          "junctions with reduced supply = %d"
          % (count, total_c, 100.0 * total_c / total if total else float("nan"), dropped))
PY
