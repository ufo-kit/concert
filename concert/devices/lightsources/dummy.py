"""Dummy lightsource implementation"""

from concert.quantities import q
from concert.devices.lightsources import base


class LightSource(base.LightSource):

    """A dummy light source"""

    def __init__(self):
        super(LightSource, self).__init__()
        self._intensity = 0 * q.V

    def _set_intensity(self, intensity):
        self._intensity = intensity

    def _get_intensity(self):
        return self._intensity
