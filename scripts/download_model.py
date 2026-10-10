#!/usr/bin/env python3
"""Download the supplied GGUF at an immutable revision, verifying its LFS SHA-256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from urllib.parse import quote, urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dida_app.local_model import MODEL_NAME

REPOSITORY = 'JonathanColetti/Qwen3.8-27B-Uncensored-GGUF'


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected


def download_ranges(url, path, size, get=requests.get, workers=4, chunk_size=256 * 1024 * 1024, prefix=None):
    """Write validated, disjoint HTTP ranges; whole-file SHA-256 is checked by caller."""
    copied = 0
    with path.open('r+b') as output:
        output.truncate(size)
        if prefix is not None:
            source, length = prefix
            while copied < length:
                data = source.read(min(8 * 1024 * 1024, length - copied))
                if not data:
                    raise ValueError('O trecho reutilizado terminou antes do tamanho esperado')
                output.write(data)
                copied += len(data)
            source.close()
    if copied > size:
        raise ValueError('Trecho reutilizado maior que o modelo')
    progress = [copied]
    progress_lock = threading.Lock()
    cancelled = threading.Event()
    def transfer(start, end):
        expected_range = f'bytes {start}-{end - 1}/{size}'
        with get(url, headers={'Range': f'bytes={start}-{end - 1}'}, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get('Content-Range') != expected_range:
                raise ValueError('O CDN não confirmou o intervalo solicitado')
            received = 0
            with path.open('r+b') as output:
                output.seek(start)
                for data in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if cancelled.is_set():
                        return
                    if not data:
                        continue
                    received += len(data)
                    if received > end - start:
                        raise ValueError('Intervalo recebido excedeu o tamanho anunciado')
                    output.write(data)
                    with progress_lock:
                        old = progress[0]
                        progress[0] += len(data)
                        if progress[0] // (128 * 1024 * 1024) != old // (128 * 1024 * 1024):
                            print(f'Download: {100 * progress[0] / size:.1f}%', flush=True)
            if received != end - start:
                raise ValueError('Intervalo incompleto')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transfer, start, min(start + chunk_size, size))
                   for start in range(copied, size, chunk_size)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
    if progress[0] != size:
        raise ValueError('Download por intervalos incompleto')


def download(destination=None, get=requests.get, prefix=None):
    destination = Path(destination or os.environ.get('DIDA_MODEL_PATH', ROOT / '.models' / MODEL_NAME))
    if not destination.is_absolute():
        destination = ROOT / destination
    if destination.name != MODEL_NAME:
        raise ValueError('O destino deve conservar o nome ' + MODEL_NAME)
    with get(f'https://huggingface.co/api/models/{REPOSITORY}', params={'blobs': 'true'}, timeout=(10, 30)) as response:
        response.raise_for_status()
        metadata = response.json()
    revision = metadata.get('sha', '')
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('O provedor não informou uma revisão imutável válida')
    file = next((row for row in metadata.get('siblings', []) if row.get('rfilename') == MODEL_NAME), None)
    lfs = (file or {}).get('lfs', {})
    expected = lfs.get('sha256', '')
    size = lfs.get('size')
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('SHA-256 LFS ausente; download interrompido sem desativar verificação')
    if not isinstance(size, int) or size < 4:
        raise ValueError('Tamanho do modelo não confirmado')
    if destination.exists():
        if destination.stat().st_size == size and verify(destination, expected):
            print('PASS: modelo existente confere com o SHA-256 do provedor')
            return destination
        raise ValueError('Já existe um arquivo diferente no destino; ele foi preservado')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < size + 1024**3:
        raise OSError('Espaço insuficiente para o modelo e a margem de 1 GiB')
    url = f'https://huggingface.co/{REPOSITORY}/resolve/{revision}/{quote(MODEL_NAME)}'
    fd, staging = tempfile.mkstemp(prefix='qwen_', suffix='.partial', dir=destination.parent)
    count = 0
    digest = hashlib.sha256()
    try:
        if size > 128 * 1024 * 1024:
            os.close(fd)
            download_ranges(url, Path(staging), size, get=get, prefix=prefix)
            count = Path(staging).stat().st_size
            with open(staging, 'rb') as file:
                for data in iter(lambda: file.read(8 * 1024 * 1024), b''):
                    digest.update(data)
        else:
            with os.fdopen(fd, 'wb') as output:
                with get(url, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        count += len(chunk)
                        if count > size:
                            raise ValueError('O download excedeu o tamanho anunciado')
                        output.write(chunk)
                        digest.update(chunk)
                        if count // (128 * 1024 * 1024) != (count - len(chunk)) // (128 * 1024 * 1024):
                            print(f'Download: {100 * count / size:.1f}%', flush=True)
        if count != size or digest.hexdigest() != expected:
            raise ValueError('Tamanho ou SHA-256 do GGUF não confere')
        with open(staging, 'rb') as file:
            if file.read(4) != b'GGUF':
                raise ValueError('Cabeçalho GGUF inválido')
        # Link publishes without overwriting a file created concurrently.
        os.link(staging, destination)
        print(f'PASS: modelo verificado, revisão {revision}, SHA-256 {expected}')
        return destination
    finally:
        Path(staging).unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        download()
    except (OSError, ValueError, requests.RequestException) as error:
        if isinstance(error, requests.RequestException):
            request = getattr(error, 'request', None)
            host = urlsplit(request.url).hostname if request is not None else 'Hugging Face'
            message = f'Falha HTTPS ({type(error).__name__}) em {host}; verifique a regra de rede.'
        else:
            message = str(error)
        print('Falha ao preparar o modelo: ' + message, file=sys.stderr)
        raise SystemExit(1)
