#!/bin/bash -l
cd "${LOGDIR:-$HOME/logs}"
now=$(date +%s)
echo "cluster time now: $(date -Is)"
for window in 3600 7200 21600; do
  n=0
  for f in rho2.o42836.*; do
    e=$(grep -m1 '=== finished' "$f" 2>/dev/null | awk '{print $3}')
    [ -z "$e" ] && continue
    es=$(date -d "$e" +%s 2>/dev/null) || continue
    [ $(( now - es )) -lt $window ] && n=$(( n + 1 ))
  done
  echo "  tasks finished in the last $(( window / 3600 ))h: $n"
done
echo
echo "--- case 2 task runtimes (recent) ---"
for t in 150 155 158; do
  f=rho2.o42836.$t
  [ -f "$f" ] || continue
  s=$(grep -m1 '=== started' "$f" | awk '{print $3}')
  e=$(grep -m1 '=== finished' "$f" | awk '{print $3}')
  [ -n "$s" ] && [ -n "$e" ] && echo "  task $t: $(( ($(date -d "$e" +%s) - $(date -d "$s" +%s)) / 60 )) min"
done
