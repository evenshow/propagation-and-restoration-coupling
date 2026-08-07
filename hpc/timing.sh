#!/bin/bash -l
cd "${LOGDIR:-$HOME/logs}"
for job in c1gate1k.o40602 c2gate1k.o40604; do
  echo "--- $job : 10 replications x 4 arms per task ---"
  for t in 1 5 40 80; do
    f="${job}.${t}"
    [ -f "$f" ] || continue
    s=$(grep -m1 '=== started' "$f" | awk '{print $3}')
    e=$(grep -m1 '=== finished' "$f" | awk '{print $3}')
    if [ -n "$s" ] && [ -n "$e" ]; then
      ss=$(date -d "$s" +%s); es=$(date -d "$e" +%s)
      echo "    task $t: $(( (es - ss) / 60 )) min $(( (es - ss) % 60 )) s"
    fi
  done
done
