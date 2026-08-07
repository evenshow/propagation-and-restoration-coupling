#!/bin/bash -l
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
export MPLBACKEND=Agg
cd "${PROJECT_ROOT:-$HOME/coupling0729}"
for spec in "layouts 1 case1_layouts" "layouts 2 case2_layouts" "cross 1 case1_cross" \
            "horizon 2 case2_horizon" "multi_T 1 case1_multiT" "multi_T 2 case2_multiT"; do
  set -- $spec
  python3 examples/run_robustness.py --mode "$1" --case "$2" --merge --outdir "results/$3"
done
