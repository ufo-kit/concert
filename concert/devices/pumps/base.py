"""Pumps."""

from concert.base import Quantity, State, check, AccessorNotImplementedError
from concert.coroutines.base import background
from concert.quantities import q
from concert.devices.base import Device


class Pump(Device):

    """A pumping device."""

    state = State(default='standby')
    flow_rate = Quantity(q.l / q.s, help="Flow rate")

    async def __ainit__(self):
        await super(Pump, self).__ainit__()

    @background
    @check(source='standby', target='pumping')
    async def start(self):
        """
        start()

        Start pumping.
        """
        await self._start()

    @background
    @check(source='pumping', target='standby')
    async def stop(self):
        """
        stop()

        Stop pumping.
        """
        await self._stop()

    async def _get_flow_rate(self):
        raise AccessorNotImplementedError

    async def _set_flow_rate(self, flow_rate):
        raise AccessorNotImplementedError

    async def _start(self):
        raise NotImplementedError

    async def _stop(self):
        raise NotImplementedError
