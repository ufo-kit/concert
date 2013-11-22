"""Shutter Dummy."""
from concert.devices.shutters import base


class Shutter(base.Shutter):

    """Shutter class implementation."""

    def __init__(self):
        super(Shutter, self).__init__()
        self._dummy_state = 'open'

    def _open(self):
        self._dummy_state = 'open'

    def _close(self):
        self._dummy_state = 'closed'
