"""A dummy gripper."""

from concert.devices.grippers import base


class Gripper(base.Gripper):

    """A dummy gripper."""

    def __init__(self):
        super(Gripper, self).__init__()
        self._state = 'released'

    async def _release(self):
        self._state = 'released'

    async def _grip(self):
        self._state = 'gripped'

    async def _get_state(self):
        return self._state
