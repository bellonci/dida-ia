"""Configuration and launch commands for a local llama.cpp GGUF server."""
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlsplit

MODEL_NAME = 'Qwen3.8-27B-Uncensored-IQ4_XS.gguf'
SUPPORTED_FILES = {MODEL_NAME, 'Qwen3.8-27B-Uncensored-Q4_K_M.gguf'}


@dataclass(frozen=True)
class LocalModelConfig:
    model_path: Path
    base_url: str = 'http://127.0.0.1:8081/v1'
    context_size: int = 8192
    threads: int = 4
    gpu_layers: int = 0

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('DIDA_LLM_BASE_URL deve apontar para um servidor HTTP local')
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip('/') != '/v1':
            raise ValueError('URL local inválida; use http://127.0.0.1:8081/v1')
        if self.context_size < 2048 or self.threads < 1 or self.gpu_layers < -1:
            raise ValueError('Configuração de contexto, threads ou GPU inválida')
        if self.model_path.name not in SUPPORTED_FILES:
            raise ValueError('O arquivo configurado deve se chamar ' + MODEL_NAME)

    @property
    def model_name(self):
        return self.model_path.name

    @classmethod
    def from_environment(cls):
        root = Path(__file__).resolve().parents[1]
        path = Path(os.environ.get('DIDA_MODEL_PATH', str(root / '.models' / MODEL_NAME))).expanduser()
        if not path.is_absolute():
            path = root / path
        return cls(path,
                   os.environ.get('DIDA_LLM_BASE_URL', 'http://127.0.0.1:8081/v1').rstrip('/'),
                   int(os.environ.get('DIDA_CONTEXT_SIZE', '8192')),
                   int(os.environ.get('DIDA_MODEL_THREADS', str(min(4, os.cpu_count() or 1)))),
                   int(os.environ.get('DIDA_GPU_LAYERS', '0')))

    def command(self, executable=None):
        root = Path(__file__).resolve().parents[1]
        executable = executable or os.environ.get('DIDA_LLAMA_SERVER') or shutil.which('llama-server')
        if not executable:
            executable = str(root / '.tools' / 'llama.cpp' / 'build' / 'bin' / 'llama-server')
        if not self.model_path.is_file():
            raise FileNotFoundError('Arquivo GGUF não encontrado: ' + str(self.model_path))
        with self.model_path.open('rb') as file:
            if file.read(4) != b'GGUF':
                raise ValueError('O arquivo não tem um cabeçalho GGUF válido')
        if not Path(executable).is_file() and not shutil.which(executable):
            raise FileNotFoundError('llama-server não instalado; execute scripts/install_llama.sh')
        parsed = urlsplit(self.base_url)
        return [executable, '--model', str(self.model_path), '--alias', self.model_name,
                '--host', parsed.hostname, '--port', str(parsed.port or 80),
                '--ctx-size', str(self.context_size), '--threads', str(self.threads),
                '--n-gpu-layers', str(self.gpu_layers), '--parallel', '1']


def start_model(config, catalog, timeout=180):
    # Preserve a pre-existing server; never terminate an unrelated process.
    if catalog.refresh(force=True):
        return None
    process = subprocess.Popen(config.command())
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('llama-server encerrou antes de carregar o modelo')
            if catalog.refresh(force=True):
                return process
            time.sleep(0.5)
        raise TimeoutError('O modelo não carregou dentro do prazo')
    except BaseException:
        stop_model(process)
        raise


def stop_model(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
