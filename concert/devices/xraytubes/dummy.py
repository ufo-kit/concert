"""A dummy X-ray tube."""

from concert.base import transition
from concert.devices.xraytubes import base
from concert.quantities import q


class XRayTube(base.XRayTube):

    """A dummy X-ray tube implementation."""

    async def __ainit__(self):
        await super(XRayTube, self).__ainit__()
        self._voltage = 0 * q.V
        self._current = 0 * q.A

    @transition(target='on')
    async def _on(self):
        pass

    @transition(target='off')
    async def _off(self):
        pass

    async def _get_voltage(self):
        return self._voltage

    async def _set_voltage(self, voltage):
        self._voltage = voltage

    async def _get_current(self):
        return self._current

    async def _set_current(self, current):
        self._current = current
