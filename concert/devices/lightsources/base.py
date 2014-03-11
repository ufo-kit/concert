"""Light sources"""
from concert.quantities import q
from concert.base import Quantity
from concert.devices.base import Device
from concert.base import AccessorNotImplementedError


class LightSource(Device):

    """A base LightSource class."""

    intensity = Quantity(q.V)

    def __init__(self):
        super(LightSource, self).__init__()

    def _set_intensity(self, value):
        raise AccessorNotImplementedError

    def _get_intensity(self):
        raise AccessorNotImplementedError
