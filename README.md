# Dida IA — Qwen local

Assistente em português com uma interface renovada, temas claro e escuro,
layout adaptável ao celular, anexos, histórico e conversa por streaming.
O script `Ia-Completa-Hibrida.py` continua sendo a entrada da aplicação.
Os componentes estão em `dida_app/`; preserve essa pasta junto ao script.

## Modelo escolhido

Toda geração de respostas, análise de documentos e redação de curiosidades usa
**Qwen local em GGUF**, servido por llama.cpp. Não há fallback de geração para
Groq, GPT ou Ollama, e nenhuma chave de API é necessária para conversar.

O arquivo do [link fornecido](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-IQ4_XS.gguf?download=true)
é **Qwen3.8-27B-Uncensored-IQ4_XS.gguf**. Essa é a variante padrão desta versão.
O nome originalmente solicitado, `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`, corresponde
a outra quantização: caso tenha esse arquivo verdadeiro, configure
`DIDA_MODEL_PATH` com seu caminho. A aplicação aceita esses dois nomes e usa o
nome real do arquivo como alias. Não renomeie uma quantização para aparentar outra.

Metadados do arquivo IQ4_XS confirmados no Hugging Face:

- Revisão: `45d0fc0ad6cfcf9eeee6008ddaea925536409b18`.
- Tamanho: `15309039008` bytes, aproximadamente 14,26 GiB.
- SHA-256: `53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5`.

O downloader consulta os metadados atuais, fixa a revisão para o download,
confere tamanho, SHA-256 e cabeçalho GGUF e preserva arquivos existentes.
Arquivos grandes são transferidos com quatro conexões e intervalos HTTP validados.
TLS e verificações de integridade permanecem ativos.
Se a configuração anterior ainda exigir `GROQ_API_KEY`, remova esse requisito
pelas configurações do ambiente: o aplicativo não o utiliza mais.

## Instalação na nuvem ou Linux

Python 3.12, GCC/G++ e ferramentas de compilação são necessários.
Use o checkout existente; cada tarefa na nuvem já é isolada, sem necessidade
de criar worktrees.

```sh
cd /workspace/dida-ia
bash scripts/install.sh
.venv/bin/python scripts/download_model.py
```

A instalação prepara a virtualenv e compila `llama-server` somente para CPU,
com quatro jobs, no cache ignorado `.tools/llama.cpp`. A revisão do llama.cpp
está fixada em `aa94f2086147cd4bfbcff818ce998dffce5fcb39`.
Os scripts não alteram automaticamente arquivos de configuração pessoal nem
baixam o GGUF durante a instalação das dependências.

O download exige acesso a `huggingface.co` e aos domínios usados pelos
redirecionamentos, incluindo `us.aws.cdn.hf.co`. Essas regras foram salvas no
rascunho da configuração da nuvem. Salvar o rascunho não comprova que a regra
já está aplicada à máquina; verifique o download depois de salvar as configurações.

O arquivo IQ4_XS foi baixado e verificado com o SHA-256 acima, carregado pelo
llama.cpp e validado com uma resposta real pelo comando `--check-model`.
O catálogo confirmou o alias correto, contexto de 8192 tokens e entrada de texto.
O fluxo real da Dida por HTTP/SSE respondeu `pronto` e persistiu a conversa;
essa solicitação levou 33,2 segundos, sem representar um benchmark geral.
Reserve espaço para os aproximadamente 15,3 GB de pesos, memória de execução e
KV cache. Nesta máquina, o servidor usa quatro threads de CPU; a latência depende
do hardware. Imagens exigem o projetor multimodal separado, que não foi instalado.

## Executar

Depois de obter o GGUF:

```sh
cd /workspace/dida-ia
DIDA_CONFIG_DIR=/workspace/dida-ia/.dida .venv/bin/python Ia-Completa-Hibrida.py --headless --start-model --port 8765
```

`--start-model` usa o arquivo configurado, aguarda o catálogo confirmar seu alias
e inicia o servidor llama.cpp em loopback. Se um servidor com o mesmo modelo já
responder, ele é reutilizado. Ao sair, a Dida encerra somente o servidor que iniciou.

Para verificar uma resposta real, em outro terminal:

```sh
.venv/bin/python Ia-Completa-Hibrida.py --check-model
```

Para desenvolver a interface e testar o aplicativo enquanto prepara os pesos:

```sh
DIDA_CONFIG_DIR=/workspace/dida-ia/.dida .venv/bin/python Ia-Completa-Hibrida.py --headless --port 8765
```

Nesse caso a interface funciona, mas informa que o modelo não está carregado
quando você envia uma mensagem. Isso não valida inferência.

O servidor da Dida é local. Valide internamente GET `/` e GET `/api/profile`.
A nuvem não tem uma prévia pública configurada. No computador com interface
gráfica, use `--browser --start-model` para abrir o navegador.

## Preferências e hardware

Use `.env.example` como referência. Variáveis já exportadas têm prioridade sobre
`.env`. Caminhos relativos de `DIDA_MODEL_PATH` são resolvidos em relação à pasta
do projeto. `DIDA_LLM_BASE_URL` aceita somente HTTP em loopback com caminho `/v1`.
`DIDA_CONTEXT_SIZE`, `DIDA_MODEL_THREADS` e `DIDA_GPU_LAYERS` controlam o servidor
iniciado pela Dida. O build na nuvem é para CPU; GPU exige um build adequado ao
hardware, indicado por `DIDA_LLAMA_SERVER`.

Perfil, memória e preferências ficam em `DIDA_CONFIG_DIR`, cujo padrão é
`~/.config/Dida`. Em **Histórico e memória**, você pode consultar as 50 conversas
mais recentes, apagar todo o histórico ou desativar gravação e recuperação de
memória. Desativar mantém os registros existentes. Apagar cancela respostas em
andamento e remove também backups de memória inválida gerados pela aplicação.
O perfil “Sobre mim” é separado do histórico.

Tavily é opcional para busca web e curiosidades baseadas em fontes. Sem sua
chave, a pesquisa usa a alternativa DuckDuckGo quando disponível. As consultas
web são enviadas ao serviço de busca; a geração usa o modelo local.

Voz e reconhecimento de fala são funções distintas do modelo de conversa:

```sh
.venv/bin/python -m pip install -r requirements-desktop.txt
```

- Microfone: requer dispositivos, PortAudio e um modelo faster-whisper já
  baixado, indicado por `DIDA_STT_MODEL_PATH`. A Dida não baixa modelos de
  transcrição implicitamente nem usa transcrição Groq/Google.
- Voz: Google/Microsoft precisam de internet; a voz do sistema usa pyttsx3 e
  vozes instaladas. Ajustes de velocidade podem exigir FFmpeg.
- Janela nativa: pywebview precisa de um backend gráfico compatível.
- Imagens: o servidor precisa confirmar suporte multimodal. O nome GGUF sozinho
  não comprova visão. Sem essa capacidade, a aplicação rejeita imagens com uma
  mensagem clara. PDFs escaneados podem exigir OCR externo ou conteúdo em texto.

Esses recursos de hardware não foram validados fisicamente na nuvem.

## Verificações

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python Ia-Completa-Hibrida.py --autoteste
.venv/bin/python -m pip check
```

Os testes incluem cancelamento durante a espera por cabeçalhos e durante o
streaming, proteção contra retomada de voz antiga, isolamento de clientes SSE,
limites de contexto, memória, anexos e checksum. O teste de fluxo completo usa um
servidor de modelo explicitamente simulado e não comprova inferência do GGUF real.

Com Playwright e Chromium disponíveis, teste a interface em desktop e celular:

```sh
# Em um terminal, instância isolada sem modelo disponível:
DIDA_CONFIG_DIR=$(mktemp -d) DIDA_LLM_BASE_URL=http://127.0.0.1:18081/v1 .venv/bin/python Ia-Completa-Hibrida.py --headless --port 8766
# Em outro terminal:
DIDA_TEST_URL=http://127.0.0.1:8766 node tests/browser-smoke.cjs
```

Esse teste usa o servidor da Dida sem modelo carregado. Ele cobre envio, temas,
diálogos, memória e confirmação de exclusão; imagens resultantes ficam em
`artifacts/`. A interface não depende de fontes ou bibliotecas em CDNs externos.

## Organização

| Componente | Responsabilidade |
| --- | --- |
| `Ia-Completa-Hibrida.py` | Configuração e coordenação do assistente |
| `dida_app/ui.html` | Interface e interações |
| `dida_app/server.py` | Rotas HTTP e escrita de eventos por cliente |
| `dida_app/requests.py` | Cancelamento por pedido; mensagem mais recente prevalece |
| `dida_app/events.py` | Filas limitadas de eventos, sem escrita de socket no broadcast |
| `dida_app/audio.py` | Síntese e reprodução com proteção contra retomada |
| `dida_app/memory.py` / `storage.py` | Histórico e persistência privada e atômica |
| `dida_app/attachments.py` / `context.py` | Leitura e seleção de trechos relevantes |
| `dida_app/local_model.py` / `models.py` | Inicialização e validação do modelo local |
| `dida_app/llm.py` / `transport.py` | Geração e transporte HTTP cancelável |

O orçamento textual usa uma estimativa conservadora por bytes UTF-8, não o
tokenizador nativo do GGUF. Documentos são divididos e classificados pela pergunta;
quando não há termos relevantes, são amostrados começo, meio e fim. Trechos têm
identificação de posição e indicação de que o documento não foi enviado inteiro.
A extração ainda tem limites por formato; não equivale a revisar cada página de
um documento grande.
