'''
Created on Apr 12, 2013

@author: farago
'''
from concert.devices.shutters.base import Shutter


class DummyShutter(Shutter):
    def __init__(self):
        super(DummyShutter, self).__init__()
        self._opened = None
        self.close().wait()

    def is_open(self):
        return self._opened

    def _open(self):
        self._opened = True

    def _close(self):
        self._opened = False
