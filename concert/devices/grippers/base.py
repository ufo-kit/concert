"""A gripper can grip and release objects."""

from concert.casync import casync
from concert.base import check, State
from concert.devices.base import Device


class Gripper(Device):

    """Base gripper class."""

    state = State(default='released')

    @casync
    @check(source='gripped', target='released')
    def release(self):
        """
        release()

        Release an object.
        """
        self._release()

    @casync
    @check(source='released', target='gripped')
    def grip(self):
        """
        grip()

        Grip an object.
        """
        self._grip()

    def _release(self):
        """The actual release implementation."""
        raise NotImplementedError

    def _grip(self):
        """The actual grip implementation."""
        raise NotImplementedError
