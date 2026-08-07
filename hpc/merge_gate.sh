#!/bin/bash -l
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT:-$HOME/coupling0729}"

python3 examples/run_case1_factorial.py --merge --replications 1000 \
    --gate-indicator infrastructure --outdir results/case1_gate_infra_n1000
python3 examples/run_case2_factorial.py --merge --replications 1000 \
    --gate-indicator infrastructure --outdir results/case2_gate_infra_n1000
