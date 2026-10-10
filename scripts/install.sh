#!/usr/bin/env bash
set -euo pipefail
dida_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dida_root"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
bash scripts/install_llama.sh
.venv/bin/python -m unittest discover -s tests -v
DIDA_CONFIG_DIR="$dida_root/.dida" .venv/bin/python Ia-Completa-Hibrida.py --autoteste
# Model download is a separate explicit command because it requires large storage and egress.
