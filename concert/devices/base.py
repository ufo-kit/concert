import threading
import logging
from concert.async import async
from concert.base import Parameterizable


LOG = logging.getLogger(__name__)


class Device(Parameterizable):

    """
    A :class:`.Device` provides locked access to a real-world device.

    It implements the context protocol to provide locking::

        with device:
            # device is locked
            device.parameter = 1 * q.m
            ...

        # device is unlocked again
    """

    def __init__(self):
        # We have to create the lock early on because it will be accessed in
        # any add_parameter calls, especially those in the Parameterizable base
        # class
        self._lock = threading.Lock()
        super(Device, self).__init__()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()

    @async
    def abort(self):
        """Emergency stop."""
        self._abort()

    def _abort(self):
        """The actual abort implementation."""
        pass


def abort(devices):
    """Abort all actions related with parameters on all *devices*."""
    futures = []
    for device in devices:
        future = device.abort()
        futures.append(future)

    return futures
