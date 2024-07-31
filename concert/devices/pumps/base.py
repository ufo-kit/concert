"""Pumps."""
from abc import abstractmethod
from concert.base import Quantity, State, check
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

    @abstractmethod
    async def _get_flow_rate(self):
        ...

    @abstractmethod
    async def _set_flow_rate(self, flow_rate):
        ...

    @abstractmethod
    async def _start(self):
        ...

    @abstractmethod
    async def _stop(self):
        ...
