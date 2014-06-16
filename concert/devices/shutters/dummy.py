"""Shutter Dummy."""
from concert.base import transition
from concert.devices.shutters import base


class Shutter(base.Shutter):

    """A dummy shutter that can be opened and closed."""

    def __init__(self):
        super(Shutter, self).__init__()

    @transition(target='open')
    def _open(self):
        pass

    @transition(target='closed')
    def _close(self):
        pass
