#!/bin/bash -l
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
cd "${PROJECT_ROOT:-$HOME/coupling0729}"
python3 examples/run_gate_threshold.py --mode horizon --merge --outdir results/case1_gate_horizon
