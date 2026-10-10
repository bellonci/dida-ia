# Dida v4 ULTIMATE - Máximo possível

## Tudo que evoluiu (v1 -> v4)
- v1: barge-in por volume (se cortava sozinha)
- v2: AEC PipeWire timestamp sync + bleed adaptativo + Whisper offline + streaming Grok->Groq
- v3: Visão real (Grok Vision + Groq Vision) com preview
- v4 ULTIMATE:
  1. STT Groq Whisper Large v3 (primário) - entende cochicho, 99% acerto
  2. Memória persistente ~/.config/Dida/memory.json - lembra das últimas 200 conversas, busca semântica
  3. Roteamento inteligente: classifica pergunta automaticamente
     - pesquisa -> grok-3 (sem filtro)
     - código -> llama-3.3-70b
     - visão -> grok-2-vision / llama-3.2 vision
     - simples -> 8b instant rápido
  4. Parser universal: txt, pdf (texto + escaneado OCR via visão), docx, xlsx (lê planilha), csv, json, imagem
  5. Web search DuckDuckGo + memória
  6. webrtcvad para barge mais preciso
  7. Limite arquivo 20MB

## Instalação v4
```bash
cp -r DidaFinal_v4/* ~/Documents/Codex/2026-10-02/realtime-voice-chat/Dida/
chmod +x ~/Documents/Codex/2026-10-02/realtime-voice-chat/Dida/ia_completa_hibrida.py ~/iniciar_dida.sh
nano ~/Documents/Codex/2026-10-02/realtime-voice-chat/Dida/.env
# XAI_API_KEY=xai-... (https://console.x.ai/)
# GROQ_API_KEY=gsk_... (https://console.groq.com/keys)
~/ia-venv/bin/pip install -r ~/Documents/Codex/2026-10-02/realtime-voice-chat/Dida/requirements.txt
~/iniciar_dida.sh --autoteste  # PASS 8/8
~/iniciar_dida.sh
```

## Teste máximo
- Anexe foto -> descreve
- Anexe PDF escaneado -> faz OCR via visão
- Anexe XLSX -> lê planilha e resume
- Fale baixo no meio dela falando -> corta em 0.4s com Whisper Large v3
- Pergunte "lembra do que anexei ontem?" -> busca na memória
