"""
.. exception:: KillException

    Exception that may be thrown during the execution of an :func:`.casync`
    decorated function. The function may run cleanup code.

.. function:: casync

    A decorator for functions which are executed asynchronously.

.. function:: threaded

    Threaded execution of a function *func*.

"""
import queue
import time
import functools
import threading
import traceback
import concert.config
from concurrent.futures import ThreadPoolExecutor, Future
from concert.quantities import q


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


def no_casync(func):
    @functools.wraps(func)
    def _inner(*args, **kwargs):
        future = NoFuture(None)
        try:
            result = func(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            if concert.config.PRINT_NOASYNC_EXCEPTION:
                traceback.print_exc()
            future.set_exception(e)

        return future

    return _inner


# Module-wide executor
EXECUTOR = ThreadPoolExecutor(max_workers=128)


# This is a stub exception that will never be raised.
class KillException(Exception):
    pass


def casync(func):
    if concert.config.ENABLE_ASYNC:
        @functools.wraps(func)
        def _inner(*args, **kwargs):
            return EXECUTOR.submit(func, *args, **kwargs)

        return _inner
    else:
        return no_casync(func)


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


def resolve(result, unit_output=True):
    """
    Generate tuples *[(x_1, y_1, ...), (x_2, y_2, ...)]* from a process that returns a list of
    futures each resulting in a single tuple *(x_1, y_1, ...)*. Set *unit_output* to ``False``
    for obtaining values without units.
    """
    for f in result:
        if unit_output:
            yield f.result()
        else:
            lst = list(f.result())
            lst = [x.magnitude for x in lst]
            yield tuple(lst)


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
