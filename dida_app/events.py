"""Bounded per-client queues; publishing never performs socket I/O."""
import queue
import threading


class Subscription:
    def __init__(self, capacity):
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = threading.Event()

    def receive(self, timeout=0.2):
        return self.queue.get(timeout=timeout)


class EventHub:
    def __init__(self, capacity=128, max_clients=32):
        self.capacity = capacity
        self.max_clients = max_clients
        self._clients = set()
        self._lock = threading.Lock()

    def subscribe(self):
        with self._lock:
            if len(self._clients) >= self.max_clients:
                raise ValueError("Muitas conexões de eventos abertas")
            client = Subscription(self.capacity)
            self._clients.add(client)
            return client

    def unsubscribe(self, client):
        with self._lock:
            self._clients.discard(client)
            client.closed.set()

    def publish(self, event):
        with self._lock:
            for client in tuple(self._clients):
                try:
                    client.queue.put_nowait(event.copy())
                except queue.Full:
                    # Reconnect rather than silently lose a partial chat response.
                    self._clients.remove(client)
                    client.closed.set()

    def close(self):
        with self._lock:
            for client in self._clients:
                client.closed.set()
            self._clients.clear()
