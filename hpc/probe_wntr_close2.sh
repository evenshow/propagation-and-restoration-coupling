#!/bin/bash -l
#
# Find a way to close a pipe that actually works on wntr 1.0.0.
# Run as: bash -l $HOME/probe_wntr_close2.sh

set -u
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":${PYTHONPATH:-}

python3 - <<'PY'
import traceback

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


def junction_total(wn):
    res = wntr.sim.WNTRSimulator(wn).run_sim()
    frame = res.node["demand"]
    if len(frame) == 0:
        return None
    row = frame.iloc[0]
    return sum(float(row.get(j, 0.0)) for j in wn.junction_name_list)


base = junction_total(build())
print("baseline junction demand = %.6f" % base)
print()

wn = build()
pipe = wn.get_link(wn.pipe_name_list[0])
print("Pipe attrs of interest:", [a for a in dir(pipe) if "status" in a.lower()])
print()

N = 40


def attempt(label, mutate):
    wn = build()
    try:
        for name in wn.pipe_name_list[:N]:
            mutate(wn, name)
        total = junction_total(wn)
    except Exception as exc:
        print("  %-38s EXCEPTION %s: %s" % (label, type(exc).__name__, exc))
        return
    if total is None:
        print("  %-38s -> INFEASIBLE (empty result)" % label)
        return
    pct = 100.0 * total / base
    verdict = "NO EFFECT" if abs(pct - 100.0) < 0.01 else "WORKS"
    print("  %-38s -> %8.4f  (%6.2f%% of baseline)  %s" % (label, total, pct, verdict))


def m_initial(wn, name):
    wn.get_link(name).initial_status = LinkStatus.Closed


def m_initial_reset(wn, name):
    wn.get_link(name).initial_status = LinkStatus.Closed
    wn.reset_initial_values()


def m_private_status(wn, name):
    link = wn.get_link(name)
    link.initial_status = LinkStatus.Closed
    object.__setattr__(link, "_user_status", LinkStatus.Closed)


def m_control(wn, name):
    link = wn.get_link(name)
    act = wntr.network.controls.ControlAction(link, "status", LinkStatus.Closed)
    cond = wntr.network.controls.SimTimeCondition(wn, "=", 0)
    ctrl = wntr.network.controls.Control(cond, act)
    wn.add_control("close_%s" % name, ctrl)


def m_remove(wn, name):
    wn.remove_link(name)


print("closing the first %d pipes by various means:" % N)
attempt("initial_status = Closed", m_initial)
attempt("initial_status + reset_initial_values", m_initial_reset)
attempt("initial_status + _user_status", m_private_status)
attempt("time-0 control setting status", m_control)
attempt("remove_link", m_remove)
PY
