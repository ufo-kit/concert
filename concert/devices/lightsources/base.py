"""Light sources"""
from concert.quantities import q
from concert.base import Quantity
from concert.devices.base import Device
from concert.base import AccessorNotImplementedError


class LightSource(Device):

    """A base LightSource class."""

    intensity = Quantity(q.V)

    async def __ainit__(self):
        await super(LightSource, self).__ainit__()

    async def _set_intensity(self, value):
        raise AccessorNotImplementedError

    async def _get_intensity(self):
        raise AccessorNotImplementedError
