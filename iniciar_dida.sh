#!/bin/bash
set -e
DIDA_DIR="$HOME/Área de trabalho/Dida IA"
DIDA_PYTHON="$HOME/ia-venv/bin/python"
if [ ! -x "$DIDA_PYTHON" ]; then
    echo "Python da Dida não encontrado. Confira o ambiente ia-venv."
    exit 1
fi
cd "$DIDA_DIR"
echo "=== Dida 5.4 Ultimate ==="
exec "$DIDA_PYTHON" "$DIDA_DIR/Ia-Completa-Hibrida.py" "$@"
