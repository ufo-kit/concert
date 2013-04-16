'''
Created on Apr 11, 2013

@author: farago
'''
from concert.devices.base import Device
from concert.asynchronous import async


class Shutter(Device):
    """Shutter device."""
    def __init__(self):
        super(Shutter, self).__init__()

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

    def is_open(self):
        """Return *True* if shutter is open, otherwise *False*."""
        raise NotImplementedError

    def _open(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError
