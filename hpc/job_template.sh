#!/bin/bash -l
#
# Generic SGE jobscript for coupling0729 on UCL Myriad.
#
# The command to run is passed after the script name, and the resource
# directives below are only defaults -- anything given on the qsub line wins:
#
#   qsub hpc/job_template.sh python3 -m pytest tests -q
#   qsub -N sweep -l h_rt=4:0:0 -l mem=8G -pe smp 16 \
#        hpc/job_template.sh python3 examples/run_dose_response.py --mode psi --replications 30
#
# Note mem is PER SLOT: -l mem=8G -pe smp 16 asks for 128G in total.
#
#$ -N coupling
#$ -l h_rt=0:30:00
#$ -l mem=4G
#$ -pe smp 4
# Cluster paths are placeholders. Either edit the three SGE directives
# below, or override them when submitting:
#   qsub -wd <project> -o <logdir> -e <logdir> ...
#$ -wd /home/USERNAME/coupling0729
#$ -o /home/USERNAME/logs/
#$ -e /home/USERNAME/logs/

module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":${PYTHONPATH:-}

# Match BLAS threads to the slots SGE actually granted. Left alone, numpy
# spawns one thread per physical core and oversubscribes the node.
export OMP_NUM_THREADS=${NSLOTS:-1}
export OPENBLAS_NUM_THREADS=${NSLOTS:-1}
export MKL_NUM_THREADS=${NSLOTS:-1}

# Compute nodes have no display; matplotlib must not try to open one.
export MPLBACKEND=Agg

echo "=== started  $(date -Is)  host=$(hostname)  NSLOTS=$NSLOTS  JOB_ID=$JOB_ID"
echo "=== workdir  $(pwd)"
echo "=== command  $*"
echo

"$@"
rc=$?

echo
echo "=== finished $(date -Is)  rc=$rc"
exit $rc
