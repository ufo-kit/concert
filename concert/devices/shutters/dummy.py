"""Shutter Dummy."""
from concert.devices.shutters import base


class Shutter(base.Shutter):

    """Shutter class implementation."""

    def __init__(self):
        super(Shutter, self).__init__()

    def _open(self):
        self._set_state(self.OPEN)

    def _close(self):
        self._set_state(self.CLOSED)
