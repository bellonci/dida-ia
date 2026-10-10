"""Validate the exact configured GGUF model on the local inference server."""
import threading
import time
import requests


class ModelCatalog:
    def __init__(self, config, ttl=30, get=None):
        self.config = config
        self.ttl = ttl
        self.get = get or requests.get
        self._lock = threading.RLock()
        self._model = None
        self._checked = 0
        self._retry_at = 0
        self._error = None

    def refresh(self, force=False):
        with self._lock:
            now = time.monotonic()
            if not force and (now < self._retry_at or (self._model and now - self._checked < self.ttl)):
                return self._model is not None
            try:
                response = self.get(self.config.base_url + '/models', timeout=(2, 5))
                try:
                    response.raise_for_status()
                    rows = response.json().get('data', [])
                    if not isinstance(rows, list):
                        raise ValueError('Catálogo inválido')
                    model = next((row for row in rows if isinstance(row, dict)
                                  and row.get('id') == self.config.model_name), None)
                    if model is None:
                        raise ValueError('O servidor não carregou o modelo configurado')
                finally:
                    response.close()
                self._model = model
                self._checked = now
                self._retry_at = 0
                self._error = None
                return True
            except (requests.RequestException, ValueError, AttributeError) as error:
                self._model = None
                self._error = (str(error) if isinstance(error, ValueError)
                               else 'Servidor do modelo local indisponível')
                self._retry_at = now + 3
                return False

    def supports_vision(self):
        if not self.refresh():
            return False
        with self._lock:
            # Do not infer multimodal support from a GGUF filename.
            architecture = self._model.get('architecture', {})
            modes = self._model.get('input_modalities', architecture.get('input_modalities', []) if isinstance(architecture, dict) else [])
            return isinstance(modes, list) and 'image' in modes

    def input_budget(self, output_tokens):
        with self._lock:
            row = self._model or {}
            meta = row.get('meta') if isinstance(row.get('meta'), dict) else {}
            context = row.get('context_window', meta.get('n_ctx', self.config.context_size))
            if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
                context = self.config.context_size
            return max(64, min(8192, context - output_tokens - 256))

    def status(self):
        available = self.refresh()
        return {'available': available, 'model': self.config.model_name,
                'error': self._error, 'vision': self.supports_vision() if available else False}
