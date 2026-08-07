#!/bin/bash -l
#
# SGE array jobscript: one task per replication block.
#
#   qsub -N c1 -t 1-50 -l h_rt=2:0:0 -l mem=4G \
#        hpc/array_job.sh examples/run_case1_factorial.py \
#        --replications 1000 --outdir results/case1_n1000
#
# $SGE_TASK_ID is 1-based and --block is 0-based, so the mapping is
# block = SGE_TASK_ID - 1 and n_blocks = the array length, passed as SGE_ARRAY_N
# because SGE does not expose the array size to the task.
#
# Deliberately single-slot: the work is embarrassingly parallel across blocks,
# so one core per task and many tasks beats few fat tasks, and it keeps BLAS
# from oversubscribing.

#$ -N c0729array
#$ -l h_rt=2:0:0
#$ -l mem=4G
# Cluster paths are placeholders. Either edit the three SGE directives
# below, or override them when submitting:
#   qsub -wd <project> -o <logdir> -e <logdir> ...
#$ -wd /home/USERNAME/coupling0729
#$ -o /home/USERNAME/logs/
#$ -e /home/USERNAME/logs/

set -u

module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":${PYTHONPATH:-}

# One core per task: several hundred tasks each spawning a full BLAS thread pool
# would thrash the nodes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

SCRIPT="$1"
shift

BLOCK=$((SGE_TASK_ID - 1))
NBLOCKS="${SGE_ARRAY_N:?SGE_ARRAY_N must be exported via qsub -v}"

echo "=== started  $(date -Is)  host=$(hostname)  task=$SGE_TASK_ID/$NBLOCKS  JOB_ID=$JOB_ID"
echo "=== script   $SCRIPT"
echo "=== block    $BLOCK of $NBLOCKS"
echo "=== args     $*"
echo

python3 "$SCRIPT" --block "$BLOCK" --n-blocks "$NBLOCKS" "$@"
rc=$?

echo
echo "=== finished $(date -Is)  rc=$rc"
exit $rc
