"""
Messaging
=========

The backbone of the local messaging system is a dispatching mechanism based on
the publish-subscribe analogy. Once a dispatcher object is created, objects can
:meth:`Dispatcher.subscribe` to messages from other objects and be notified
when other objects :meth:`Dispatcher.send` a message to the dispatcher::

    from concert.asynchronous import Dispatcher

    def handle_message(sender):
        print("{0} send me a message".format(sender))

    dispatcher = Dispatcher()

    obj = {}
    dispatcher.subscribe(obj, 'foo', handle_message)
    dispatcher.send(obj, 'foo')

Concurrency
===========

Every user defined function or method **must** be synchronous (blocking).
Asynchronous execution is provided by Concert using the *async* decorator.
Every asynchronous function returns an instance of *Future* class, which can
be used for explicit synchronization. The asynchronous execution provided by
Concert deals with concurrency. If the user wants to employ real parallelism
they should make use of the multiprocessing module which provides functionality
not limited by Python's global interpreter lock.
"""
import threading
import Queue
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
        self._messages = Queue.Queue()
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
