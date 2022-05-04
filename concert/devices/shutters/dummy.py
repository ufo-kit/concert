"""Shutter Dummy."""
from concert.base import transition
from concert.devices.shutters import base


class Shutter(base.Shutter):

    """A dummy shutter that can be opened and closed."""

    async def __ainit__(self):
        await super(Shutter, self).__ainit__()

    @transition(target='open')
    async def _open(self):
        pass

    @transition(target='closed')
    async def _close(self):
        pass
