"""A gripper can grip and release objects."""

from concert.async import async
from concert.base import check, State
from concert.devices.base import Device


class Gripper(Device):

    """Base gripper class."""

    state = State(default='released')

    @async
    @check(source='gripped', target='released')
    def release(self):
        """
        release(self)

        Release an object.
        """
        self._release()

    @async
    @check(source='released', target='gripped')
    def grip(self):
        """
        grip(self)

        Grip an object.
        """
        self._grip()

    def _release(self):
        """The actual release implementation."""
        raise NotImplementedError

    def _grip(self):
        """The actual grip implementation."""
        raise NotImplementedError
