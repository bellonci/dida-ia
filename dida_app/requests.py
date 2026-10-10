"""Latest message wins, with independent cancellation for every response."""
from collections import namedtuple
import queue
import threading

Request = namedtuple('Request', 'message image mime attachment token')


class CancellationToken:
    def __init__(self):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._resources = set()

    def cancelled(self):
        return self.event.is_set()

    def register(self, resource):
        with self._lock:
            if not self.cancelled():
                self._resources.add(resource)
                return True
        resource.close()
        return False

    def unregister(self, resource):
        with self._lock:
            self._resources.discard(resource)

    def cancel(self):
        with self._lock:
            if self.cancelled():
                return
            self.event.set()
            resources = tuple(self._resources)
            self._resources.clear()
        # Response.close can wait for a socket read. Never block the submitter.
        for resource in resources:
            threading.Thread(target=self._close, args=(resource,), daemon=True).start()

    @staticmethod
    def _close(resource):
        try:
            resource.close()
        except Exception:
            pass


class RequestCoordinator:
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)
        self.lock = threading.RLock()
        self.current = None

    def _discard_pending(self):
        while True:
            try:
                pending = self.queue.get_nowait()
            except queue.Empty:
                break
            pending.token.cancel()
            self.queue.task_done()

    def submit(self, message, image=None, mime=None, attachment=None, on_replace=None):
        with self.lock:
            if self.current is not None:
                self.current.cancel()
            self._discard_pending()
            if on_replace:
                on_replace()
            token = CancellationToken()
            self.current = token
            request = Request(message, image, mime, attachment, token)
            self.queue.put_nowait(request)
            return token

    def activate(self, request, on_activate=None):
        with self.lock:
            if self.current is not request.token or request.token.cancelled():
                return False
            if on_activate:
                on_activate()
            return True

    def run_if_current(self, token, action):
        with self.lock:
            if token is not self.current or token.cancelled():
                return False
            action()
            return True

    def cancel_all(self):
        with self.lock:
            if self.current:
                self.current.cancel()
            self._discard_pending()
