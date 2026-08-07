#!/bin/bash -l
# Severity calibration for Section 3.1: how the attribution moves with flood_peak.
# One array per peak so the existing runner and array script are reused unchanged.
cd "${PROJECT_ROOT:-$HOME/coupling0729}"
for peak in 0.40 0.50 0.60 0.70 0.80 0.90; do
  tag=$(echo "$peak" | tr -d '.')
  qsub -N sev${tag} -t 1-10 -l h_rt=2:0:0 -l mem=4G -v SGE_ARRAY_N=10 \
       hpc/array_job.sh examples/run_case1_factorial.py \
       --replications 100 --flood-peak "$peak" \
       --outdir "results/case1_severity/peak_${tag}"
done
