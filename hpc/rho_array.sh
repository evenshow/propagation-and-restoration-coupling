#!/bin/bash -l
#
# SGE array jobscript for the rho_int design sweep. One task = one design point
# and one slice of that point's replications, so a task fits a modest wall-clock
# request and a failure costs a slice rather than a whole grid cell.
#
#   qsub -N rho1 -t 1-200 -l h_rt=3:0:0 -l mem=4G -v SGE_ARRAY_N=200 \
#        hpc/rho_array.sh --case 1 --replications 200 --outdir results/case1_rho_sweep

# Cluster paths are placeholders. Either edit the three SGE directives
# below, or override them when submitting:
#   qsub -wd <project> -o <logdir> -e <logdir> ...
#$ -wd /home/USERNAME/coupling0729
#$ -o /home/USERNAME/logs/
#$ -e /home/USERNAME/logs/

set -u

module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

TASK=$((SGE_TASK_ID - 1))
NTASKS="${SGE_ARRAY_N:?SGE_ARRAY_N must be exported via qsub -v}"

echo "=== started  $(date -Is)  host=$(hostname)  task=$SGE_TASK_ID/$NTASKS  JOB_ID=$JOB_ID"
echo "=== args     $*"
echo

python3 examples/run_rho_sweep.py --task "$TASK" --n-tasks "$NTASKS" "$@"
rc=$?

echo
echo "=== finished $(date -Is)  rc=$rc"
exit $rc
