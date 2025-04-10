"""A gripper can grip and release objects."""
from abc import abstractmethod

from concert.base import check, State
from concert.coroutines.base import background
from concert.devices.base import Device


class Gripper(Device):

    """Base gripper class."""

    state = State(default='released')

    @background
    @check(source='gripped', target='released')
    async def release(self):
        """
        release()

        Release an object.
        """
        await self._release()

    @background
    @check(source='released', target='gripped')
    async def grip(self):
        """
        grip()

        Grip an object.
        """
        await self._grip()

    @abstractmethod
    async def _release(self):
        """The actual release implementation."""
        ...

    @abstractmethod
    async def _grip(self):
        """The actual grip implementation."""
        ...
