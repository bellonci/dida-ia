"""Local HTTP streaming whose socket can be cancelled before response headers."""
import http.client
import json
import socket
import threading
from urllib.parse import urlsplit

import requests


class HTTPStream:
    def __init__(self, connection):
        self.connection = connection
        self.socket = connection.sock
        self.response = None
        self._closed = threading.Event()

    def raise_for_status(self):
        if not 200 <= self.response.status < 300:
            raise requests.HTTPError(f'Local model returned HTTP {self.response.status}')

    def iter_lines(self, chunk_size=1):
        while not self._closed.is_set():
            line = self.response.readline(1024 * 1024 + 1)
            if len(line) > 1024 * 1024:
                raise ValueError('Evento do modelo excedeu 1 MB')
            if not line:
                return
            yield line.rstrip(b'\r\n')

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        # Shutdown wakes a thread waiting in getresponse/readline; close alone may not.
        try:
            if self.socket is not None:
                self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
        if self.response is not None:
            self.response.close()


def open_stream(url, payload, token=None):
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1','localhost','::1'}:
        raise ValueError('Streaming permitido somente no servidor local')
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=3)
    stream = None
    try:
        connection.connect()
        connection.sock.settimeout(120)
        stream = HTTPStream(connection)
        if token is not None and not token.register(stream):
            return None
        connection.request('POST', parsed.path, body=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                           headers={'Content-Type':'application/json','Accept':'text/event-stream'})
        stream.response = connection.getresponse()
        return stream
    except BaseException:
        if stream is not None:
            if token is not None:
                token.unregister(stream)
            stream.close()
        else:
            connection.close()
        raise
