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
        params = [Parameter('exposure-time',
                            self._get_exposure_time,
                            self._set_exposure_time,
                            q.s),
                  Parameter('roi-width',
                            self._get_roi_width),
                  Parameter('roi-height',
                            self._get_roi_height)]

        super(DummyCamera, self).__init__(params)
        self._exposure_time = 1 * q.ms

        if background:
            self._width = background.shape[1]
            self._height = background.shape[0]
            self._background = background
        else:
            shape = (640, 480)
            self._width, self._height = shape
            self._background = np.ones(shape)

    def _get_exposure_time(self):
        return self._exposure_time

    def _set_exposure_time(self, time):
        self._exposure_time = time

    def _get_roi_width(self):
        return self._width

    def _get_roi_height(self):
        return self._height

    def _record_real(self):
        pass

    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass

    def _grab_real(self):
        noise = np.random.poisson(1.0, (self._width, self._height))
        time = self._exposure_time.rescale(q.s).magnitude
        return self._background + time * noise
