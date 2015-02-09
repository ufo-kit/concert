"""
.. exception:: KillException

    Exception that may be thrown during the execution of an :func:`.async`
    decorated function. The function may run cleanup code.

.. function:: async

    A decorator for functions which are executed asynchronously.

.. function:: threaded

    Threaded execution of a function *func*.

"""
import time
import functools
import concert.config
from concurrent.futures import ThreadPoolExecutor, Future
from concert.quantities import q

try:
    import Queue as queue
except ImportError:
    import queue


# Patch futures so that they provide a join() and kill() method
def _join(self, _timeout=None):
    try:
        self._old_result()
    except KeyboardInterrupt:
        self.cancel()

    return self


def _result(self, timeout=None):
    try:
        return self._old_result(timeout=timeout)
    except KeyboardInterrupt:
        self.cancel()


def _cancel(self):
    try:
        self._old_cancel()
    finally:
        if self.cancel_operation:
            self.cancel_operation()


def _kill(self, exception=None, block=True, timeout=None):
    pass


Future._old_cancel = Future.cancel
Future.cancel = _cancel
Future._old_result = Future.result
Future.result = _result
Future.join = _join
Future.kill = _kill
Future.cancel_operation = None


class NoFuture(Future):

    def __init__(self, result, cancel_operation=None):
        super(NoFuture, self).__init__()
        self.set_result(result)
        self.cancel_operation = cancel_operation


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


if concert.config.ENABLE_GEVENT:
    try:
        import gevent
        import gevent.monkey
        import gevent.threadpool

        gevent.monkey.patch_all()

        # XXX: we have to import threading after patching
        import threading

        HAVE_GEVENT = True
        KillException = gevent.GreenletExit
        threadpool = gevent.threadpool.ThreadPool(4)

        class GreenletFuture(gevent.Greenlet):

            """A Future interface based on top of a Greenlet.

            This class provides the :class:`concurrent.futures.Future` interface on
            top of a Greenlet.
            """

            def __init__(self, func, args, kwargs, cancel_operation=None):
                super(GreenletFuture, self).__init__()
                self.func = func
                self.args = args
                self.kwargs = kwargs
                self.saved_exception = None
                self._cancelled = False
                self._running = False
                self.cancel_operation = cancel_operation

            def _run(self, *args, **kwargs):
                try:
                    self._running = True
                    value = self.func(*self.args, **self.kwargs)
                    self._running = False

                    # Force starting at least a bit of the greenlet
                    gevent.sleep(0)
                    return value
                except Exception as exception:
                    self.saved_exception = exception

            def join(self, timeout=None):
                try:
                    super(GreenletFuture, self).join(timeout)
                except KeyboardInterrupt:
                    self.cancel()

                if self.saved_exception:
                    raise self.saved_exception

            def cancel(self):
                self._cancelled = True
                try:
                    self.kill()
                finally:
                    if self.cancel_operation:
                        self.cancel_operation()

                return True

            def cancelled(self):
                return self._cancelled

            def running(self):
                return self._running

            def done(self):
                return self.ready()

            def result(self, timeout=None):
                try:
                    value = self.get(timeout=timeout)
                except KeyboardInterrupt:
                    self.cancel()

                if self.saved_exception:
                    raise self.saved_exception

                return value

            def add_done_callback(self, callback):
                self.link(callback)

        def async(func):
            if concert.config.ENABLE_ASYNC:
                @functools.wraps(func)
                def _inner(*args, **kwargs):
                    g = GreenletFuture(func, args, kwargs)
                    g.start()
                    return g

                return _inner
            else:
                return no_async(func)

        def threaded(func):
            @functools.wraps(func)
            def _inner(*args, **kwargs):
                result = threadpool.spawn(func, *args, **kwargs)
                return result

            return _inner
    except ImportError:
        HAVE_GEVENT = False
        print("Gevent is not available, falling back to threads")
else:
    HAVE_GEVENT = False


if not concert.config.ENABLE_GEVENT or not HAVE_GEVENT:
        import threading

        # Module-wide executor
        EXECUTOR = ThreadPoolExecutor(max_workers=128)

        # This is a stub exception that will never be raised.
        class KillException(Exception):
            pass

        def async(func):
            if concert.config.ENABLE_ASYNC:
                @functools.wraps(func)
                def _inner(*args, **kwargs):
                    return EXECUTOR.submit(func, *args, **kwargs)

                return _inner
            else:
                return no_async(func)

        def threaded(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                """Execute in a separate thread."""
                thread = threading.Thread(target=func, args=args, kwargs=kwargs)
                thread.daemon = True
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


def resolve(result):
    """
    Generate tuples *[(x_1, y_1, ...), (x_2, y_2, ...)]* from a process that returns a list of
    futures each resulting in a single tuple *(x_1, y_1, ...)*.
    """
    for f in result:
        yield f.result()


class WaitError(Exception):

    """Raised on busy waiting timeouts"""
    pass


def busy_wait(condition, sleep_time=1e-1 * q.s, timeout=None):
    """Busy wait until a callable *condition* returns True. *sleep_time* is the time to sleep
    between consecutive checks of *condition*. If *timeout* is given and the *condition* doesn't
    return True within the time specified by it a :class:`.WaitingError` is raised.
    """
    sleep_time = sleep_time.to(q.s).magnitude
    if timeout:
        start = time.time()
        timeout = timeout.to(q.s).magnitude

    while not condition():
        if timeout and time.time() - start > timeout:
            raise WaitError('Waiting timed out')
        time.sleep(sleep_time)
