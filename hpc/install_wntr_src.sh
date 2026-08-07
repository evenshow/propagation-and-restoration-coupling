#!/bin/bash -l
#
# Rebuild wntr against the numpy that Myriad actually has.
# Run as: bash -l $HOME/install_wntr_src.sh
#
# The wheel for wntr 1.0.0 carries a compiled evaluator built against numpy
# C-API 0x10 (numpy >= 1.22). Myriad's bundle is numpy 1.21.5 (0xe), so the
# extension imports fine as a file and then fails at
# `numpy.core.multiarray failed to import`.
#
# `--no-binary` forces a source build so the extension compiles against the
# installed numpy. `--no-build-isolation` is the load-bearing flag: with build
# isolation pip creates a throwaway environment and installs a *fresh, newer*
# numpy to build against, which reproduces exactly the incompatibility we are
# trying to remove.

set -u
module load python3/recommended
TARGET="${PYLIBS:-$HOME/pylibs}"
export PYTHONPATH=$TARGET:${PYTHONPATH:-}

echo "=== numpy on the build path"
python3 -c "import numpy; print(numpy.__version__, numpy.get_include())"
echo

# Remove the incompatible wheel's artefacts first, or the source build may be
# skipped as already-satisfied.
rm -rf "$TARGET"/wntr "$TARGET"/wntr-*.dist-info
echo "=== removed previous wntr from $TARGET"
echo

for version in 1.0.0 0.5.0; do
    echo "=== attempting source build: wntr==$version"
    if python3 -m pip install --target "$TARGET" --no-deps --upgrade \
            --no-binary :all: --no-build-isolation "wntr==$version" 2>&1 | tail -5; then
        if PYTHONPATH=$TARGET python3 -c "import wntr.sim" 2>/dev/null; then
            echo "    BUILT AND IMPORTS: wntr==$version"
            break
        fi
        echo "    built but still fails to import; trying next version"
        rm -rf "$TARGET"/wntr "$TARGET"/wntr-*.dist-info
    else
        echo "    build failed for $version"
        rm -rf "$TARGET"/wntr "$TARGET"/wntr-*.dist-info
    fi
    echo
done

echo
echo "=== verification"
python3 - <<'PY'
import importlib

for name in ("wntr", "wntr.network", "wntr.sim", "wntr.sim.aml"):
    try:
        module = importlib.import_module(name)
        print("  OK      %-18s %s" % (name, getattr(module, "__version__", "")))
    except Exception as exc:
        print("  BROKEN  %-18s %s: %s" % (name, type(exc).__name__, exc))
        raise SystemExit(1)

import wntr

wn = wntr.network.WaterNetworkModel("Net3")
wn.options.time.duration = 0
wn.options.time.hydraulic_timestep = 3600
wn.options.hydraulic.demand_model = "PDD"
results = wntr.sim.WNTRSimulator(wn).run_sim()
demand = results.node["demand"]
print("wntr Net3: %d junctions, %d pipes, rows=%d, total demand=%.6f"
      % (len(wn.junction_name_list), len(wn.pipe_name_list),
         len(demand), float(demand.iloc[0].sum())))
print("WNTR SMOKE OK")
PY
