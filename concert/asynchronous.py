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
def _wait(self, _timeout=None):
    self.result()
    return self

Future.wait = _wait

# Module-wide executor
EXECUTOR = ThreadPoolExecutor(max_workers=128)

# Module-wide disable
DISABLE = False


class _FakeFuture(Future):

    def __init__(self, result):
        super(_FakeFuture, self).__init__()
        self.set_result(result)


def async(func):
    """A decorator for functions which are supposed to be executed
    asynchronously."""

    if DISABLE:
        @wraps(func)
        def _sync(*args, **kwargs):
            result = func(*args, **kwargs)
            return _FakeFuture(result)

        _sync.__dict__["_async"] = True
        return _sync
    else:
        @wraps(func)
        def _async(*args, **kwargs):
            return EXECUTOR.submit(func, *args, **kwargs)

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
        tsm = sender, message
        if tsm in self._subscribers:
            self._subscribers[tsm].add(handler)
        else:
            self._subscribers[tsm] = set([handler])

    def unsubscribe(self, sender, message, handler):
        """Remove *handler* from the subscribers to *(sender, message)*."""
        tsm = sender, message
        if tsm in self._subscribers:
            self._subscribers[tsm].remove(handler)

    def send(self, sender, message):
        """Send message from sender."""
        self._messages.put((sender, message))

    def _serve(self):
        while True:
            tsm = self._messages.get()
            sender, message = tsm

            if tsm in self._subscribers:
                for callback in self._subscribers[tsm]:
                    callback(sender)

            if tsm in self._event_queues:
                self._event_queues[tsm].notify_and_clear()

            self._messages.task_done()


dispatcher = Dispatcher()
