#!/bin/bash -l
#
# Regenerate Figures 3 and 4 (four-arm resilience curves + recovery survival
# panel). Must run here rather than locally: the first three panels re-execute
# the engine, so they depend on the installed pandapower/wntr versions and have
# to match the numbers the tables were computed from.
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_ROOT:-$HOME/coupling0729}"

python3 examples/plot_resilience_curves.py --case 1 \
    --indir results/case1_n1000 --out results/fig3_case1_curves.png
python3 examples/plot_resilience_curves.py --case 2 \
    --indir results/case2_n1000 --out results/fig4_case2_curves.png
