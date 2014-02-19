"""This module provides a simple dummy camera."""
import numpy as np
from concert.quantities import q
from concert.base import Quantity
from concert.devices.cameras import base
import os
import time
from concert.storage import read_image


class Base(base.Camera):

    exposure_time = Quantity(unit=q.s)
    sensor_pixel_width = Quantity(unit=q.micrometer)
    sensor_pixel_height = Quantity(unit=q.micrometer)
    roi_x0 = Quantity(unit=q.pixel)
    roi_y0 = Quantity(unit=q.pixel)
    roi_width = Quantity(unit=q.pixel)
    roi_height = Quantity(unit=q.pixel)

    def __init__(self):
        super(Base, self).__init__()
        self._frame_rate = 1000 / q.s
        self._trigger_mode = self.trigger_modes.AUTO
        self._exposure_time = 1 * q.ms
        self._roi_x0 = 0 * q.pixel
        self._roi_y0 = 0 * q.pixel
        self._roi_width = 640 * q.pixel
        self._roi_height = 480 * q.pixel

    def _get_sensor_pixel_width(self):
        return 5 * q.micrometer

    def _get_sensor_pixel_height(self):
        return 5 * q.micrometer

    def _get_roi_x0(self):
        return self._roi_x0

    def _set_roi_x0(self, x0):
        self._roi_x0 = x0

    def _get_roi_y0(self):
        return self._roi_y0

    def _set_roi_y0(self, y0):
        self._roi_y0 = y0

    def _get_roi_width(self):
        return self._roi_width

    def _set_roi_width(self, roi):
        self._roi_width = roi

    def _get_roi_height(self):
        return self._roi_height

    def _set_roi_height(self, roi):
        self._roi_height = roi

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


class Camera(Base):

    """Simple camera.

    *background* can be an array-like that will be used to generate the frame
    when calling :meth:`.grab`. The final image will be the background +
    poisson noise depending on the currently set exposure time.

   """

    def __init__(self, background=None):
        super(Camera, self).__init__()

        if background is not None:
            self.roi_width = background.shape[1] * q.pixel
            self.roi_height = background.shape[0] * q.pixel
            self._background = background
        else:
            shape = (640, 480)
            self.roi_width, self.roi_height = shape * q.pixel
            self._background = np.ones(shape)

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


class FileCamera(Base):

    """A camera that reads files in a *directory*."""

    def __init__(self, directory):
        # Let users change the directory
        self.directory = directory
        super(FileCamera, self).__init__()

        self.roi_width = 0 * q.pixel
        self.roi_height = 0 * q.pixel
        self._index = 0
        self._files = [os.path.join(directory, file_name) for file_name in
                       sorted(os.listdir(directory))]

    def _grab_real(self):
        if self._index < len(self._files):
            image = read_image(self._files[self._index])

            if not self.roi_height:
                y_region = image.shape[0]
            else:
                y_region = self.roi_y0 + self.roi_height

            if not self.roi_width:
                x_region = image.shape[1]
            else:
                x_region = self.roi_x0 + self.roi_width

            result = image[self.roi_y0.magnitude:y_region, self.roi_x0.magnitude:x_region]
            self._index += 1
        else:
            result = None

        return result


class BufferedCamera(Camera, base.BufferedMixin):

    def _readout_real(self):
        for i in range(3):
            yield self.grab()
