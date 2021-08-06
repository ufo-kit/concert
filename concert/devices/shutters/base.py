"""Shutter Device."""
from concert.base import check, State, AccessorNotImplementedError
from concert.commands import command
from concert.devices.base import Device


class Shutter(Device):

    """Shutter device class implementation."""

    state = State(default="open")

    def __init__(self):
        super(Shutter, self).__init__()

    @command(name='sopen')
    @check(source='closed', target='open')
    async def open(self):
        """open()

        Open the shutter."""
        await self._open()

    @command(name='sclose')
    @check(source='open', target='closed')
    async def close(self):
        """close()

        Close the shutter."""
        await self._close()

    async def _open(self):
        raise AccessorNotImplementedError

    async def _close(self):
        raise AccessorNotImplementedError
