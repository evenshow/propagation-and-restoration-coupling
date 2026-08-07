#!/bin/bash -l
#
# Add pandapower + wntr to $HOME/pylibs. Run as: bash -l $HOME/install_deps.sh
#
# Versions are pinned deliberately. Left unpinned, pip resolves to current
# releases that require a newer pandas/numpy than the python3/recommended bundle
# provides, then tries to build pandas from source with meson+cython, which fails
# against this toolchain. `--no-deps` keeps the resolver away from numpy, scipy
# and pandas entirely; anything those packages genuinely need and the bundle does
# not already have is listed here explicitly.
#
# Consequence to keep in mind: these are older than the local versions
# (pandapower 3.5.2, wntr 1.5.0), so every number that goes in the paper must be
# produced in ONE environment. That is why the full run happens here, not locally.

set -u
module load python3/recommended
TARGET="${PYLIBS:-$HOME/pylibs}"
export PYTHONPATH=$TARGET:${PYTHONPATH:-}

echo "=== target: $TARGET"
echo "=== python: $(which python3)  $(python3 -V 2>&1)"
echo

# deepdiff/tqdm/ordered-set are pandapower's own requirements; wntr needs
# nothing beyond what the bundle ships.
PKGS="
pandapower==2.13.1
deepdiff==6.7.1
ordered-set==4.1.0
tqdm==4.66.1
wntr==1.0.0
"

for pkg in $PKGS; do
    echo "--- installing $pkg"
    python3 -m pip install --quiet --target "$TARGET" --no-deps --upgrade "$pkg" \
        && echo "    ok" \
        || echo "    FAILED: $pkg"
done

echo
echo "=== verifying imports (a --target install can truncate subpackages silently)"
python3 - <<'PY'
import importlib

checks = [
    "pandapower",
    "pandapower.networks",
    "pandapower.auxiliary",
    "wntr",
    "wntr.network",
    "wntr.sim",
    "deepdiff",
    "tqdm",
]
bad = []
for name in checks:
    try:
        module = importlib.import_module(name)
        print("  OK      %-24s %s" % (name, getattr(module, "__version__", "")))
    except Exception as exc:
        bad.append(name)
        print("  BROKEN  %-24s %s: %s" % (name, type(exc).__name__, exc))

print()
if bad:
    print("BROKEN: %s" % " ".join(bad))
    raise SystemExit(1)

# Smoke-test the two things the cases actually do, not just the imports.
import pandapower as pp
net = pp.networks.case33bw()
pp.runpp(net, numba=False, init="auto")
print("pandapower case33bw: %d buses, %d lines, slack vm=%.4f"
      % (len(net.bus), len(net.line), float(net.res_bus.vm_pu.iloc[0])))

import wntr
wn = wntr.network.WaterNetworkModel("Net3")
wn.options.time.duration = 0
wn.options.time.hydraulic_timestep = 3600
wn.options.hydraulic.demand_model = "PDD"
results = wntr.sim.WNTRSimulator(wn).run_sim()
demand = results.node["demand"]
print("wntr Net3: %d junctions, %d pipes, demand rows=%d, total=%.4f"
      % (len(wn.junction_name_list), len(wn.pipe_name_list),
         len(demand), float(demand.iloc[0].sum())))
print("SMOKE OK")
PY
