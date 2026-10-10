"""OpenAI-compatible streaming through the configured local GGUF model only."""
import http.client
import json
import requests
from .context import limit_messages
from .transport import open_stream

UNAVAILABLE = 'Não consegui gerar uma resposta agora. Verifique se o modelo local está carregado.'


def chat_stream(app, messages, image_b64=None):
    if app.request_cancelled():
        return False
    if not app.model_catalog.refresh():
        return False
    if image_b64 and not app.model_catalog.supports_vision():
        return False
    response = None
    token = getattr(app.response_context, 'token', None)
    produced = False
    try:
        app.broadcast({'type': 'status', 'text': 'Preparando uma resposta…'})
        output_tokens = 1800 if (getattr(app.response_context, 'attachment', None) or {}).get('report') else 600
        payload = {'model': app.local_model.model_name,
                   'messages': limit_messages(messages, app.model_catalog.input_budget(output_tokens)),
                   'max_tokens': output_tokens, 'temperature': 0.6, 'stream': True,
                   'chat_template_kwargs': {'enable_thinking': False}}
        response = open_stream(app.local_model.base_url + '/chat/completions', payload, token)
        if response is None:
            return False
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if app.request_cancelled():
                return False
            if not line or not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                return produced
            try:
                item = json.loads(data)
                delta = item['choices'][0]['delta'].get('content', '')
                if isinstance(delta, str) and delta:
                    produced = True
                    yield delta
            except (ValueError, KeyError, IndexError, TypeError):
                # A malformed partial response must not be saved as a completed answer.
                raise ValueError('Evento inválido no fluxo do modelo local')
        raise ValueError('O fluxo terminou sem confirmar a conclusão')
    except (requests.RequestException, ValueError, OSError, http.client.HTTPException) as error:
        print(f'[modelo local] {type(error).__name__}')
        app.response_context.failed = True
        return False
    finally:
        if response is not None:
            if token is not None:
                token.unregister(response)
            response.close()


def complete(app, messages, max_tokens=120):
    if not app.model_catalog.refresh():
        return ''
    with requests.post(app.local_model.base_url + '/chat/completions',
                       json={'model': app.local_model.model_name,
                             'messages': limit_messages(messages, app.model_catalog.input_budget(max_tokens)),
                             'max_tokens': max_tokens, 'temperature': 0.4, 'stream': False,
                             'chat_template_kwargs': {'enable_thinking': False}},
                       timeout=(3, 120)) as response:
        response.raise_for_status()
        return response.json()['choices'][0]['message'].get('content', '')
