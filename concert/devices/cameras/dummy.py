"""This module provides a simple dummy camera."""
import numpy as np
from concert.quantities import q
from concert.base import Quantity
from concert.devices.cameras import base
import os
import time
from concert.storage import read_image


class FileCamera(base.Camera):

    """A camera that reads files in a *directory*."""

    def __init__(self, directory):
        # Let users change the directory
        self.directory = directory
        super(FileCamera, self).__init__()

        self._frame_rate = 1000 * q.count / q.s
        self._trigger_mode = self.trigger_modes.AUTO
        self._index = 0
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = None
        self.roi_height = None
        self._files = [os.path.join(directory, file_name) for file_name in
                       sorted(os.listdir(directory))]

    def _get_frame_rate(self):
        return self._frame_rate

    def _set_frame_rate(self, frame_rate):
        self._frame_rate = frame_rate

    def _get_trigger_mode(self):
        return self._trigger_mode

    def _set_trigger_mode(self, mode):
        self._trigger_mode = mode

    def _record_real(self):
        pass

    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass

    def _grab_real(self):
        if self._index < len(self._files):
            image = read_image(self._files[self._index])

            if self.roi_height is None:
                y_region = image.shape[0]
            else:
                y_region = self.roi_y + self.roi_height

            if self.roi_width is None:
                x_region = image.shape[1]
            else:
                x_region = self.roi_x + self.roi_width

            result = image[self.roi_y:y_region, self.roi_x:x_region]
            self._index += 1
        else:
            result = None

        return result


class Camera(base.Camera):

    """Simple camera.

    *background* can be an array-like that will be used to generate the frame
    when calling :meth:`.grab`. The final image will be the background +
    poisson depending on the currently set exposure time.

    .. py:attribute:: exposure_time
    .. py:attribute:: roi_width

        Width of the image returned by :meth:`.grab`.

    .. py:attribute:: roi_height

        Height of the image returned by :meth:`.grab`.

    .. py:attribute:: sensor_pixel_width
    .. py:attribute:: sensor_pixel_height
    """

    exposure_time = Quantity(unit=q.s)
    sensor_pixel_width = Quantity(unit=q.micrometer)
    sensor_pixel_height = Quantity(unit=q.micrometer)

    def __init__(self, background=None):
        super(Camera, self).__init__()

        self._frame_rate = 10.0 / q.s
        self._trigger_mode = self.trigger_modes.AUTO
        self._exposure_time = 1 * q.ms

        if background is not None:
            self.roi_width = background.shape[1]
            self.roi_height = background.shape[0]
            self._background = background
        else:
            shape = (640, 480)
            self.roi_width, self.roi_height = shape
            self._background = np.ones(shape)

    def _get_sensor_pixel_width(self):
        return 5 * q.micrometer

    def _get_sensor_pixel_height(self):
        return 5 * q.micrometer

    def _get_exposure_time(self):
        return self._exposure_time

    def _set_exposure_time(self, value):
        self._exposure_time = value

    def _get_frame_rate(self):
        return self._frame_rate

    def _set_frame_rate(self, frame_rate):
        self._frame_rate = frame_rate

    def _get_trigger_mode(self):
        return self._trigger_mode

    def _set_trigger_mode(self, mode):
        self._trigger_mode = mode

    def _record_real(self):
        pass

    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass

    def _grab_real(self):
        start = time.time()
        cur_time = self.exposure_time.to(q.s).magnitude
        # 1e5 is a dummy correlation between exposure time and emitted e-.
        tmp = self._background + cur_time * 1e5
        max_value = np.iinfo(np.uint16).max
        tmp = np.random.poisson(tmp)
        # Cut values beyond the bit-depth.
        tmp[tmp > max_value] = max_value
        duration = time.time() - start
        to_sleep = 1.0 / self.frame_rate
        to_sleep = to_sleep.to_base_units() - duration * q.s
        if to_sleep > 0 * q.s:
            time.sleep(to_sleep.magnitude)

        return np.cast[np.uint16](tmp)
