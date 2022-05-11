"""Shutter Device."""
from concert.base import check, State, AccessorNotImplementedError
from concert.coroutines.base import background
from concert.devices.base import Device


class Shutter(Device):

    """Shutter device class implementation."""

    state = State(default="open")

    async def __ainit__(self):
        await super(Shutter, self).__ainit__()

    @background
    @check(source='closed', target='open')
    async def open(self):
        """open()

        Open the shutter."""
        await self._open()

    @background
    @check(source='open', target='closed')
    async def close(self):
        """close()

        Close the shutter."""
        await self._close()

    async def _open(self):
        raise AccessorNotImplementedError

    async def _close(self):
        raise AccessorNotImplementedError

    async def _emergency_stop(self):
        if await self.get_state() == 'open':
            await self.close()
