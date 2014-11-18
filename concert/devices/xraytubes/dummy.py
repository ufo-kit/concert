"""A dummy X-ray tube."""

from concert.base import transition
from concert.devices.xraytubes import base
from concert.quantities import q


class XRayTube(base.XRayTube):

    """A dummy X-ray tube implementation."""

    def __init__(self):
        super(XRayTube, self).__init__()
        self._voltage = 0 * q.V
        self._current = 0 * q.A

    @transition(target='on')
    def _on(self):
        pass

    @transition(target='off')
    def _off(self):
        pass

    def _get_voltage(self):
        return self._voltage

    def _set_voltage(self, voltage):
        self._voltage = voltage

    def _get_current(self):
        return self._current

    def _set_current(self, current):
        self._current = current
