"""Shutter Device."""
from concert.base import check, State, AccessorNotImplementedError
from concert.async import async
from concert.devices.base import Device
from concert.async import dispatcher


class Shutter(Device):

    """Shutter device class implementation."""

    state = State(default="open")

    def __init__(self):
        super(Shutter, self).__init__()

    @async
    @check(source='closed', target='open')
    def open(self):
        """open()

        Open the shutter."""
        self._open()
        dispatcher.send(self, "state_changed")

    @async
    @check(source='open', target='closed')
    def close(self):
        """close()

        Close the shutter."""
        self._close()
        dispatcher.send(self, "state_changed")

    def _open(self):
        raise AccessorNotImplementedError

    def _close(self):
        raise AccessorNotImplementedError

    def _abort(self):
        self._close()
