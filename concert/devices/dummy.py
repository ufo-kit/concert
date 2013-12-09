"""Dummy"""
from concert.base import Parameter
from concert.devices.base import Device
from concert.async import async


class DummyDevice(Device):

    """A dummy device."""

    value = Parameter()

    def __init__(self):
        super(DummyDevice, self).__init__()
        self._value = None

    def _get_value(self):
        """Get the real value."""
        return self._value

    def _set_value(self, value):
        """The real value setter."""
        self._value = value

    @async
    def do_nothing(self):
        """Do nothing."""
        pass
