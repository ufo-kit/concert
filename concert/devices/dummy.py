"""Dummy"""
import random
import time
from concert.base import Parameter
from concert.devices.base import Device
from concert.helpers import async


class DummyDevice(Device):

    """A dummy device."""

    def __init__(self):
        parameter = Parameter('value', self._get_value, self._set_value)
        super(DummyDevice, self).__init__([parameter])
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
