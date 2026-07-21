"""A dummy gripper."""

from concert.devices.grippers import base


class Gripper(base.Gripper):

    """A dummy gripper."""

    async def __ainit__(self, **kwargs):
        await super().__ainit__(**kwargs)
        self._state = 'released'

    async def _release(self):
        self._state = 'released'

    async def _grip(self):
        self._state = 'gripped'

    async def _get_state(self):
        return self._state
