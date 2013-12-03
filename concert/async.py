import functools
import concert.config
from concurrent.futures import ThreadPoolExecutor, Future

try:
    import Queue as queue
except ImportError:
    import queue


# Module-wide executor
EXECUTOR = ThreadPoolExecutor(max_workers=128)


# Patch futures so that they provide a join() and kill() method
def _join(self, _timeout=None):
    self.result()
    return self


def _kill(self, exception=None, block=True, timeout=None):
    pass


Future.join = _join
Future.kill = _kill


class NoFuture(Future):

    def __init__(self, result):
        super(NoFuture, self).__init__()
        self.set_result(result)


def no_async(func):
    @functools.wraps(func)
    def _inner(*args, **kwargs):
        future = NoFuture(None)
        try:
            result = func(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

        return future

    return _inner


try:
    if concert.config.DISABLE_GEVENT:
        raise ImportError

    import gevent
    import gevent.monkey

    gevent.monkey.patch_all()

    # XXX: we have to import threading after patching
    import threading

    HAVE_GEVENT = True
    KillException = gevent.GreenletExit

    class GreenletFuture(gevent.Greenlet):
        """A common Greenlet/Future interface.

        This class provides the :class:`concurrent.futures.Future` interface on
        top of a Greenlet.
        """

        def __init__(self, func, args, kwargs):
            super(GreenletFuture, self).__init__()
            self.func = func
            self.args = args
            self.kwargs = kwargs
            self.saved_exception = None

        def _run(self, *args, **kwargs):
            try:
                return self.func(*self.args, **self.kwargs)
            except Exception as exception:
                self.saved_exception = exception

        def join(self, timeout=None):
            super(GreenletFuture, self).join(timeout)

            if self.saved_exception:
                raise self.saved_exception

        def result(self):
            value = self.get()

            if self.saved_exception:
                raise self.saved_exception

            return value

        def done(self):
            return self.ready()

        def add_done_callback(self, callback):
            self.link(callback)

    def async(func):
        @functools.wraps(func)
        def _inner(*args, **kwargs):
            g = GreenletFuture(func, args, kwargs)
            g.start()
            return g

        return _inner

    def wait(futures):
        """Wait for the list of *futures* to finish and raise exceptions if
        happened."""
        for future in futures:
            future.join()

except ImportError:
    import threading
    HAVE_GEVENT = False

    # This is a stub exception that will never be raised.
    class KillException(Exception):
        pass

    def async(func):
        """A decorator for functions which are supposed to be executed
        asynchronously."""

        if concert.config.DISABLE_ASYNC:
            return no_async(func)
        else:
            @functools.wraps(func)
            def _inner(*args, **kwargs):
                return EXECUTOR.submit(func, *args, **kwargs)

            return _inner

    def wait(objects):
        gevent.wait(objects)


def threaded(func):
    """Threaded execution of a function *func*."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Execute in a separate thread."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


def wait(futures):
    """Wait for the list of *futures* to finish and raise exceptions if
    happened."""
    for future in futures:
        future.join()


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


