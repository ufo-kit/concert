import asyncio
import logging
from concert.base import Parameterizable
from concert.coroutines.base import background


LOG = logging.getLogger(__name__)


class Device(Parameterizable):

    """
    A :class:`.Device` provides locked access to a real-world device.

    It implements the context protocol to provide locking::

        async with device:
            # device is locked
            await device.set_parameter(1 * q.m)
            ...

        # device is unlocked again
    """

    async def __ainit__(self):
        # We have to create the lock early on because it will be accessed in
        # any add_parameter calls, especially those in the Parameterizable base
        # class
        self._lock = asyncio.Lock()
        await super(Device, self).__ainit__()

    async def __aenter__(self):
        await self._lock.acquire()

        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._lock.release()

    @background
    async def emergency_stop(self):
        """Emergency stop."""
        await self._emergency_stop()

    async def _emergency_stop(self):
        """Emergency stop implementation."""
        pass
