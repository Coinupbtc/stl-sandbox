#!/bin/bash
# STL Sandbox installer — idempotent. Creates .venv and installs everything,
# including the aarch64 cadquery workaround (no nlopt wheel exists; a shim
# satisfies the import — see nlopt_shim/nlopt.py).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt
./.venv/bin/pip install --quiet --no-deps "cadquery==2.8.0"
./.venv/bin/pip install --quiet "cadquery-ocp<8.0,>=7.9.3.1" "ezdxf>=1.3.0" \
  "multimethod<2.0,>=1.11" runtype casadi trame trame-vtk trame-components \
  trame-vuetify "pyparsing>=3.0.0" scipy numba

SP=$(./.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
ln -sf "$PWD/nlopt_shim/nlopt.py" "$SP/nlopt.py"

./.venv/bin/python -c "import cadquery, trimesh, fastapi; print('install OK: cadquery', cadquery.__version__)"
echo "Run with: OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python app.py"
