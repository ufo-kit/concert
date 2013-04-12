'''
Created on Apr 11, 2013

@author: farago
'''
from concert.base import Device
from concert.asynchronous import async


class Shutter(Device):
    """Shutter device."""
    @async
    def open(self):
        self._open()

    @async
    def close(self):
        self._close()

    def _open(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError
