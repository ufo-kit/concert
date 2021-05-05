"""A gripper can grip and release objects."""

from concert.commands import command
from concert.base import check, State
from concert.devices.base import Device


class Gripper(Device):

    """Base gripper class."""

    state = State(default='released')

    @command()
    @check(source='gripped', target='released')
    async def release(self):
        """
        release()

        Release an object.
        """
        await self._release()

    @command()
    @check(source='released', target='gripped')
    async def grip(self):
        """
        grip()

        Grip an object.
        """
        await self._grip()

    async def _release(self):
        """The actual release implementation."""
        raise NotImplementedError

    async def _grip(self):
        """The actual grip implementation."""
        raise NotImplementedError
