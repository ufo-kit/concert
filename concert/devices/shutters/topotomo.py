"""Topotomo"""
from concert.devices.shutters import base
from concert.connections.tango import TopoTomo
import time


class Shutter(base.Shutter):

    """Shutter class implementation."""

    def __init__(self, index):
        if index < 0 or index > 2:
            raise ValueError("Index must be in range [0-2].")

        super(Shutter, self).__init__()
        self._device = TopoTomo().get_device("iss/toto/rato_toto")
        self._index = index

    @property
    def index(self):
        """Return Index."""
        return self._index

    def _open(self):
        self._device.DoSPECTextCommand("shopen %d" % (self.index))
        time.sleep(5)
        self._set_state(self.OPEN)

    def _close(self):
        self._device.DoSPECTextCommand("shclose %d" % (self.index))
        time.sleep(5)
        self._set_state(self.CLOSED)
