"""Shutter Device."""
from concert.base import check, State, AccessorNotImplementedError
from concert.casync import casync
from concert.devices.base import Device


class Shutter(Device):

    """Shutter device class implementation."""

    state = State(default="open")

    def __init__(self):
        super(Shutter, self).__init__()

    @casync
    @check(source='closed', target='open')
    def open(self):
        """open()

        Open the shutter."""
        self._open()

    @casync
    @check(source='open', target='closed')
    def close(self):
        """close()

        Close the shutter."""
        self._close()

    def _open(self):
        raise AccessorNotImplementedError

    def _close(self):
        raise AccessorNotImplementedError

    def _abort(self):
        self._close()
