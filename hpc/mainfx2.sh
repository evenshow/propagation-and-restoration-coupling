#!/bin/bash -l
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
cd "${PROJECT_ROOT:-$HOME/coupling0729}"
python3 - <<'PY'
import pandas as pd
for case in (1, 2):
    e = pd.read_csv(f"results/case{case}_rho_sweep/effects.csv")
    m = e.groupby(["q_prop", "kappa_prop"])[
        ["loss__delta_prop", "loss__delta_rec", "loss__delta_int"]].mean().reset_index()
    print(f"\n########## CASE {case}: ratio across the threshold (kappa 0.6 -> 0.8) ##########")
    rows = []
    for q in sorted(m.q_prop.unique()):
        lo = m[(m.q_prop == q) & (m.kappa_prop == 0.6)].iloc[0]
        hi = m[(m.q_prop == q) & (m.kappa_prop == 0.8)].iloc[0]
        rows.append({
            "q": q,
            "prop_x": hi.loss__delta_prop / lo.loss__delta_prop,
            "rec_x": hi.loss__delta_rec / lo.loss__delta_rec,
            "int_x": hi.loss__delta_int / lo.loss__delta_int if lo.loss__delta_int > 1e-9 else float("inf"),
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))
PY
