"""Shutter Device."""
from concert.helpers import async
from concert.devices.base import Device


class Shutter(Device):

    """Shutter device class implementation."""
    OPEN = "open"
    CLOSED = "closed"

    def __init__(self):
        super(Shutter, self).__init__()
        self._states = self._states.union(set([self.OPEN, self.CLOSED]))
        self._state = self.NA

    @async
    def open(self):
        """open()

        Open the shutter."""
        self._open()

    @async
    def close(self):
        """close()

        Close the shutter."""
        self._close()

    def _open(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError
