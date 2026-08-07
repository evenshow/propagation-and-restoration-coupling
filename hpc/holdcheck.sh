#!/bin/bash -l
echo "--- hold reason for 42836 ---"
qstat -j 42836 2>/dev/null | grep -iE "hold|error|jobshare|task|slots" | head -20
echo
echo "--- any failed tasks so far? ---"
cd "${LOGDIR:-$HOME/logs}"
echo "rho1 non-zero: $(grep -l 'rc=[^0]' rho1.o42835.* 2>/dev/null | wc -l) of $(ls rho1.o42835.* 2>/dev/null | wc -l)"
echo "rho2 non-zero: $(grep -l 'rc=[^0]' rho2.o42836.* 2>/dev/null | wc -l) of $(ls rho2.o42836.* 2>/dev/null | wc -l)"
echo
echo "--- typical rho1 task runtime ---"
for t in 1 100 200; do
  f=rho1.o42835.$t
  [ -f "$f" ] || continue
  s=$(grep -m1 '=== started' "$f" | awk '{print $3}')
  e=$(grep -m1 '=== finished' "$f" | awk '{print $3}')
  [ -n "$s" ] && [ -n "$e" ] && echo "  task $t: $(( ($(date -d "$e" +%s) - $(date -d "$s" +%s)) / 60 )) min"
done
