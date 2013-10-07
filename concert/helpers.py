"""The :mod:`.helpers` module provides mechanisms for asynchronous
execution and messaging.
"""
try:
    import Queue as queue
except ImportError:
    import queue

import threading
from functools import wraps
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, Future


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
            future = _FakeFuture(None)
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

            return future

        _sync.__dict__["_async"] = True
        return _sync
    else:
        @wraps(func)
        def _async(*args, **kwargs):
            return EXECUTOR.submit(func, *args, **kwargs)

        _async.__dict__["_async"] = True
        return _async


def threaded(func):
    """Threaded execution of a function *func*."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Execute in a separate thread."""
        thread = Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


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


def coroutine(func):
    """
    Start a coroutine automatically without the need to call
    next() or send(None) first.
    """
    @wraps(func)
    def start(*args, **kwargs):
        """Starts the generator."""
        gen = func(*args, **kwargs)
        gen.next()
        return gen
    return start


def inject(generator, consumer):
    """
    Let a *generator* produce a value and forward it to *consumer*.
    """
    for item in generator:
        consumer.send(item)


@coroutine
def multicast(*consumers):
    """
    multicast(*consumers)

    Forward data to all *consumers*.
    """
    while True:
        item = yield
        for consumer in consumers:
            consumer.send(item)
