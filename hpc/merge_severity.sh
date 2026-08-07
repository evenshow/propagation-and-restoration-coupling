#!/bin/bash -l
module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":$PYTHONPATH
export MPLBACKEND=Agg
cd "${PROJECT_ROOT:-$HOME/coupling0729}"
for peak in 040 050 060 070 080 090; do
  p=$(echo "$peak" | sed 's/^0\(.\)\(.\)$/\1.\2/')
  python3 examples/run_case1_factorial.py --merge --replications 100 \
      --flood-peak "$p" --outdir "results/case1_severity/peak_${peak}" >/dev/null 2>&1
  echo "merged peak_${peak}"
done
python3 - <<'PY'
import pandas as pd
from pathlib import Path
rows = []
for peak in ("040", "050", "060", "070", "080", "090"):
    d = Path(f"results/case1_severity/peak_{peak}")
    s = pd.read_csv(d / "effect_summary.csv")
    runs = pd.read_csv(d / "runs.csv", dtype={"arm": str})
    runs["arm"] = runs["arm"].str.zfill(2)
    rec = {"flood_peak": float(peak) / 100}
    for metric, ch in (("loss", "delta_prop"), ("loss", "delta_rec"),
                       ("loss", "delta_int"), ("loss", "total"),
                       ("Lambda", "delta_prop"), ("lock_in", "delta_int")):
        r = s[(s.metric == metric) & (s.channel == ch)]
        rec[f"{metric}_{ch}"] = float(r.iloc[0]["mean"]) if not r.empty else float("nan")
        if not r.empty:
            rec[f"{metric}_{ch}_lo"] = float(r.iloc[0].ci_low)
            rec[f"{metric}_{ch}_hi"] = float(r.iloc[0].ci_high)
    rec["frac_damaged"] = float(runs["frac_damaged"].mean())
    rec["F_min_arm00"] = float(runs.loc[runs.arm == "00", "F_min"].mean())
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv("results/case1_severity/severity_summary.csv", index=False)
cols = ["flood_peak", "frac_damaged", "Lambda_delta_prop", "loss_delta_prop",
        "loss_delta_rec", "loss_delta_int", "loss_total", "lock_in_delta_int"]
print(df[cols].round(4).to_string(index=False))
PY
