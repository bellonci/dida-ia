#!/bin/bash
set -e
PROJ="$HOME/Documents/Codex/2026-10-02/realtime-voice-chat/Dida"
VENV="$HOME/ia-venv/bin/python"

echo "=== Instalador Dida v4 ULTIMATE COMPLETO ==="

mkdir -p "$PROJ"
mkdir -p ~/.local/share/applications
mkdir -p ~/.config/systemd/user
mkdir -p ~/.config/Dida

# Copia arquivos principais
cp ia_completa_hibrida.py "$PROJ/"
cp requirements.txt "$PROJ/"
cp iniciar_dida.sh "$HOME/iniciar_dida.sh"
chmod +x "$PROJ/ia_completa_hibrida.py" "$HOME/iniciar_dida.sh"

# .env
[ -f "$PROJ/.env" ] || cp .env.example "$PROJ/.env"
echo "-> .env em $PROJ/.env (edite suas chaves)"

# Desktop
cp dida.desktop ~/.local/share/applications/dida.desktop
chmod +x ~/.local/share/applications/dida.desktop
echo "-> Atalho no menu: Dida v4 Ultimate"

# Systemd
cp dida.service ~/.config/systemd/user/dida.service
systemctl --user daemon-reload
echo "-> Serviço criado: dida.service"
echo "   Para ativar inicialização automática:"
echo "   systemctl --user enable dida.service"
echo "   Para iniciar agora em background:"
echo "   systemctl --user start dida.service"
echo "   Para ver logs:"
echo "   journalctl --user -u dida.service -f"

# Deps
echo "-> Instalando dependências..."
$VENV -m pip install -r "$PROJ/requirements.txt" || $VENV -m pip install -r requirements.txt

echo ""
echo "=== PRONTO - V4 ULTIMATE ==="
echo "1. Edite chaves: nano $PROJ/.env"
echo "   XAI_API_KEY=xai-... (https://console.x.ai/)"
echo "   GROQ_API_KEY=gsk_... (https://console.groq.com/keys)"
echo "2. Teste: ~/iniciar_dida.sh --autoteste  (PASS 8/8)"
echo "3. Rode: ~/iniciar_dida.sh  -> http://127.0.0.1:8765"
echo "4. Ou use o menu iniciar: Dida v4 Ultimate"
echo "5. Autostart: systemctl --user enable --now dida.service"
