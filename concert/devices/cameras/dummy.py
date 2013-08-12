"""This module provides a simple dummy camera."""
import numpy as np
from concert.quantities import q
from concert.base import Parameter
from concert.devices.cameras import base


class Camera(base.Camera):

    """Simple camera.

    *background* can be an array-like that will be used to generate the frame
    when calling :meth:`.grab`. The final image will be the background +
    poisson depending on the currently set exposure time.

    :py:attribute:: exposure_time
    :py:attribute:: roi_width
    :py:attribute:: roi_height
    :py:attribute:: sensor_pixel_width
    :py:attribute:: sensor_pixel_height
    """

    def __init__(self, background=None):
        params = [Parameter('exposure-time', unit=q.s),
                  Parameter('roi-width'),
                  Parameter('roi-height'),
                  Parameter('sensor-pixel-width', unit=q.micrometer),
                  Parameter('sensor-pixel-height', unit=q.micrometer)]

        super(Camera, self).__init__(params)

        self.exposure_time = 1 * q.ms
        self.sensor_pixel_width = 5 * q.micrometer
        self.sensor_pixel_height = 5 * q.micrometer

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
        time = self.exposure_time.to(q.s).magnitude

        # 1e5 is a dummy correlation between exposure time and emitted e-.
        tmp = self._background + time * 1e5
        max_value = np.iinfo(np.uint16).max
        tmp = np.random.poisson(tmp)
        # Cut values beyond the bit-depth.
        tmp[tmp > max_value] = max_value

        return np.cast[np.uint16](tmp)
