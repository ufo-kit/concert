"""The :mod:`.asynchronous` module provides mechanisms for asynchronous
execution and messaging.
"""
try:
    import Queue as queue
except ImportError:
    import queue

import threading
from concurrent.futures import ThreadPoolExecutor, Future


executor = ThreadPoolExecutor(max_workers=10)


def _patch_futures():
    def wait(self, timeout=None):
        self.result()
        return self
    Future.wait = wait


def async(func, *args, **kwargs):
    """This function is intended to be used as a decorator for functions
    which are supposed to be executed asynchronously."""
    def _async(*args, **kwargs):
        return executor.submit(func, *args, **kwargs)

    res = _async
    res.__name__ = func.__name__
    res.__dict__["_async"] = True

    return res


def is_async(func):
    """returns *True* if the given function *func* is asynchronous."""
    return hasattr(func, "_async")


def wait(futures):
    """Wait for the list of *futures* to finish with checking exceptions."""
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
