#!/bin/bash -l
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
cd "${PROJECT_ROOT:-$HOME/coupling0729}"
python3 - <<'PY'
import pandas as pd
e = pd.read_csv("results/case1_rho_sweep/effects.csv")
g = e.groupby(["q_prop", "kappa_prop"])
m = g[["loss__delta_prop", "loss__delta_rec", "loss__delta_int"]].mean().reset_index()
for col, name in (("loss__delta_prop", "Delta_prop  (arm 10: restoration OFF)"),
                  ("loss__delta_rec",  "Delta_rec   (arm 01: propagation OFF)"),
                  ("loss__delta_int",  "Delta_int   (needs all four arms)")):
    print("\n=== %s ===" % name)
    print(m.pivot(index="q_prop", columns="kappa_prop", values=col).round(4).to_string())
PY
