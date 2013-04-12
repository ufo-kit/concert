'''
Created on Apr 12, 2013

@author: farago
'''
from concert.devices.shutters import base
from concert.connection import TangoConnection


class Shutter(base.Shutter):
    def __init__(self, index):
        if index < 0 or index > 2:
            raise ValueError("Index must be in range [0-2].")

        super(Shutter, self).__init__()
        self._connection = TangoConnection("iss/toto/rato_toto",
                                           "anka-tango", 10018)
        self._index = index

    @property
    def index(self):
        return self._index

    def _open(self):
        self._connection.device.DoSPECTextCommand("shopen %d" % (self.index))

    def _close(self):
        self._connection.device.DoSPECTextCommand("shclose %d" % (self.index))
