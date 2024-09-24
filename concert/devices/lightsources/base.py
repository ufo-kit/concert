"""Light sources"""
from abc import abstractmethod

from concert.quantities import q
from concert.base import Quantity
from concert.devices.base import Device


class LightSource(Device):

    """A base LightSource class."""

    intensity = Quantity(q.V)

    async def __ainit__(self):
        await super(LightSource, self).__ainit__()

    @abstractmethod
    async def _set_intensity(self, value):
        ...

    @abstractmethod
    async def _get_intensity(self):
        ...
