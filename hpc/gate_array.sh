#!/bin/bash -l
# Cluster paths are placeholders. Either edit the three SGE directives
# below, or override them when submitting:
#   qsub -wd <project> -o <logdir> -e <logdir> ...
#$ -wd /home/USERNAME/coupling0729
#$ -o /home/USERNAME/logs/
#$ -e /home/USERNAME/logs/
set -u
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
TASK=$((SGE_TASK_ID - 1))
NTASKS="${SGE_ARRAY_N:?SGE_ARRAY_N must be exported via qsub -v}"
echo "=== started $(date -Is) task=$SGE_TASK_ID/$NTASKS"
python3 examples/run_gate_threshold.py --task "$TASK" --n-tasks "$NTASKS" "$@"
rc=$?
echo "=== finished $(date -Is) rc=$rc"
exit $rc
