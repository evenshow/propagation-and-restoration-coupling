#!/bin/bash -l
#
# Does Myriad have what coupling0729 needs? Run as: bash -l $HOME/check_env.sh
#
# `module` is a shell function that only exists in a login shell, so this must be
# invoked with `bash -l`; a plain `bash` silently falls through to /bin/python3.

module load python3/recommended
export PYTHONPATH="${PYLIBS:-$HOME/pylibs}":${PYTHONPATH:-}

echo "python3   : $(which python3)"
python3 - <<'PY'
import importlib
import sys

print("version   : %d.%d.%d" % sys.version_info[:3])
print()
need = [
    ("numpy", "core"),
    ("scipy", "core"),
    ("pandas", "core"),
    ("networkx", "case 1 feeder layout"),
    ("pandapower", "case 1 power flow"),
    ("wntr", "case 2 hydraulics"),
    ("matplotlib", "figures (merge step only)"),
    ("pytest", "test suite"),
]
missing = []
for name, why in need:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "?")
        print("  OK      %-12s %-10s  (%s)" % (name, version, why))
    except Exception as exc:
        missing.append(name)
        print("  MISSING %-12s %-10s  (%s)  [%s]" % (name, "-", why, type(exc).__name__))

print()
print("missing: %s" % (" ".join(missing) if missing else "none"))
PY
