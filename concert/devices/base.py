import asyncio
import threading
import logging
from concert.base import Parameterizable
from concert.commands import command


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

    @command()
    async def abort(self):
        """Emergency stop."""
        await self._abort()

    async def _abort(self):
        """The actual abort implementation."""
        pass


def abort_all_devices(devices):
    """Abort all actions related with parameters on all *devices* and return gathering future from
    :func:`asyncio.gather`.
    """
    return asyncio.gather(*[device.abort() for device in devices])
