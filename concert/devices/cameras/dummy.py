"""This module provides a simple dummy camera."""
import numpy as np
import quantities as q
from concert.base import Parameter
from concert.devices.cameras.base import Camera


class DummyCamera(Camera):
    """Simple camera.

    *background* can be an array-like that will be used to generate the frame
    when calling :meth:`.grab`. The final image will be the background +
    poisson depending on the currently set exposure time.

    :py:attribute:: exposure_time
    :py:attribute:: roi_width
    :py:attribute:: roi_height
    """

    def __init__(self, background=None):
        params = [Parameter('exposure-time', unit=q.s),
                  Parameter('roi-width'),
                  Parameter('roi-height')]

        super(DummyCamera, self).__init__(params)

        self.exposure_time = 1 * q.ms

        if background:
            self.roi_width = background.shape[1]
            self.roi_height = background.shape[0]
            self._background = background
        else:
            shape = (640, 480)
            self.roi_width, self.roi_height = shape
            self._background = np.ones(shape)

    def _record_real(self):
        pass

    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass

    def _grab_real(self):
        noise = np.random.poisson(1.0, (self.roi_width, self.roi_height))
        time = self.exposure_time.rescale(q.s).magnitude
        return self._background + time * noise
