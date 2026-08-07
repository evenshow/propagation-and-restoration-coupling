#!/bin/bash -l
date -Is
echo "--- queue by job ---"
for j in 99382 99383 99384 99385 99386 99387; do
  if qstat -j $j >/dev/null 2>&1; then
    r=$(qstat 2>/dev/null | awk -v id=$j '$1==id && $5=="r"' | wc -l)
    q=$(qstat 2>/dev/null | awk -v id=$j '$1==id && $5!="r"' | wc -l)
    echo "  $j : running=$r  waiting-entries=$q"
  else
    echo "  $j : gone from queue"
  fi
done
echo
echo "--- blocks produced ---"
cd "${PROJECT_ROOT:-$HOME/coupling0729}"/results
for d in case1_layouts case2_layouts case1_cross case2_horizon case1_multiT case2_multiT; do
  n=$(ls "$d/blocks" 2>/dev/null | wc -l)
  echo "  $d : $n"
done
echo
echo "non-zero exits: $(grep -l 'rc=[^0]' ${LOGDIR:-$HOME/logs}/rob*.o* 2>/dev/null | wc -l) of $(ls ${LOGDIR:-$HOME/logs}/rob*.o* 2>/dev/null | wc -l)"
