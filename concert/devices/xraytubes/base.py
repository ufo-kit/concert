"""
An X-ray tube.
"""
from abc import abstractmethod

from concert.base import check
from concert.coroutines.base import background
from concert.quantities import q
from concert.base import Quantity, State
from concert.devices.base import Device


class XRayTube(Device):
    """
    A base x-ray tube class.
    """

    voltage = Quantity(q.kV)
    current = Quantity(q.uA)
    power = Quantity(q.W)

    state = State(default='off')

    async def __ainit__(self):
        await super(XRayTube, self).__ainit__()

    @abstractmethod
    async def _get_voltage(self):
        ...

    @abstractmethod
    async def _set_voltage(self, voltage):
        ...

    @abstractmethod
    async def _get_current(self):
        ...

    @abstractmethod
    async def _set_current(self, current):
        ...

    async def _get_power(self):
        return (await self.get_voltage() * await self.get_current()).to(q.W)

    @background
    @check(source='off', target='on')
    async def on(self):
        """
        on()

        Enables the x-ray tube.
        """
        await self._on()

    @background
    @check(source='on', target='off')
    async def off(self):
        """
        off()

        Disables the x-ray tube.
        """
        await self._off()

    @abstractmethod
    async def _on(self):
        """
        Implementation of on().
        """
        ...

    @abstractmethod
    async def _off(self):
        """
        Implementation of off().
        """
        ...
