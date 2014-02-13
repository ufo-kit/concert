"""Shutter Device."""
from concert.base import transition, State
from concert.async import async
from concert.devices.base import Device


class Shutter(Device):

    """Shutter device class implementation."""

    state = State(default="open")

    def __init__(self):
        super(Shutter, self).__init__()

    @async
    @transition(source='closed', target='open')
    def open(self):
        """open()

        Open the shutter."""
        self._open()

    @async
    @transition(source='open', target='closed')
    def close(self):
        """close()

        Close the shutter."""
        self._close()

    def _open(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError
