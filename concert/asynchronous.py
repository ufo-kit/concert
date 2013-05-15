"""The :mod:`.asynchronous` module provides mechanisms for asynchronous
execution and messaging.
"""
try:
    import Queue as queue
except ImportError:
    import queue

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps


# Patch futures so that they provide a wait() method
def _wait(self, timeout=None):
    self.result()
    return self

Future.wait = _wait

# Module-wide executor
executor = ThreadPoolExecutor(max_workers=10)


def async(func):
    """A decorator for functions which are supposed to be executed
    asynchronously."""

    @wraps(func)
    def _async(*args, **kwargs):
        return executor.submit(func, *args, **kwargs)

    _async.__dict__["_async"] = True

    return _async


def is_async(func):
    """Returns *True* if the given function *func* is asynchronous."""
    return hasattr(func, "_async") and getattr(func, "_async")


def wait(futures):
    """Wait for the list of *futures* to finish and raise exceptions if
    happened."""
    for future in futures:
        future.result()


class Dispatcher(object):
    """Core dispatcher"""

    def __init__(self):
        self._subscribers = {}
        self._messages = queue.Queue()
        self._event_queues = {}
        self._lock = threading.Lock()

        server = threading.Thread(target=self._serve)
        server.daemon = True
        server.start()

    def subscribe(self, sender, message, handler):
        """Subscribe to a message sent by sender.

        When message is sent by sender, handler is called with sender as the
        only argument.

        """
        t = sender, message
        if t in self._subscribers:
            self._subscribers[t].add(handler)
        else:
            self._subscribers[t] = set([handler])

    def unsubscribe(self, sender, message, handler):
        """Remove *handler* from the subscribers to *(sender, message)*."""
        t = sender, message
        if t in self._subscribers:
            self._subscribers[t].remove(handler)

    def send(self, sender, message):
        """Send message from sender."""
        self._messages.put((sender, message))

    def _serve(self):
        while True:
            t = self._messages.get()
            sender, message = t

            if t in self._subscribers:
                for callback in self._subscribers[t]:
                    callback(sender)

            if t in self._event_queues:
                self._event_queues[t].notify_and_clear()

            self._messages.task_done()


dispatcher = Dispatcher()
