"""Dummy lightsource implementation"""

from concert.quantities import q
from concert.devices.lightsources import base


class LightSource(base.LightSource):

    """A dummy light source"""

    async def __ainit__(self):
        await super(LightSource, self).__ainit__()
        self._intensity = 0 * q.V

    async def _set_intensity(self, intensity):
        self._intensity = intensity

    async def _get_intensity(self):
        return self._intensity
