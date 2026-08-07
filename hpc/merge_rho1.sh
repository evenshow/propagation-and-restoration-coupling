#!/bin/bash -l
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
export MPLBACKEND=Agg
cd "${PROJECT_ROOT:-$HOME/coupling0729}"
python3 examples/run_rho_sweep.py --case 1 --replications 200 --merge \
    --outdir results/case1_rho_sweep
