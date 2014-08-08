"""This module provides a simple dummy camera."""
import os
import time
import numpy as np
from concert.quantities import q
from concert.base import transition, Quantity
from concert.devices.cameras import base
from concert.storage import read_image


class Base(base.Camera):

    exposure_time = Quantity(q.s)
    sensor_pixel_width = Quantity(q.micrometer)
    sensor_pixel_height = Quantity(q.micrometer)
    roi_x0 = Quantity(q.pixel)
    roi_y0 = Quantity(q.pixel)
    roi_width = Quantity(q.pixel)
    roi_height = Quantity(q.pixel)

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

    @transition(target='recording')
    def _record_real(self):
        pass

    @transition(target='standby')
    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass


class Camera(Base):

    """A simple dummy camera."""

    def __init__(self, background=None):
        """
        *background* can be an array-like that will be used to generate the frame
        when calling :meth:`.grab`. The final image will be the background +
        poisson noise depending on the currently set exposure time.
        """
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

    """A camera that reads files in a *directory*. If *reset_on_start* is True the files are read
    from the beginning when the recording starts.
    """

    def __init__(self, directory, reset_on_start=True):
        # Let users change the directory
        self.directory = directory
        super(FileCamera, self).__init__()

        self.index = 0
        self.reset_on_start = reset_on_start
        self.filenames = [os.path.join(directory, file_name) for file_name in
                          sorted(os.listdir(directory))]

        if not self.filenames:
            raise base.CameraError("No files found")

        image = read_image(self.filenames[0])
        self._roi_width = image.shape[1] * q.pixel
        self._roi_height = image.shape[0] * q.pixel

    @transition(target='recording')
    def _record_real(self):
        if self.reset_on_start:
            self.index = 0

    def _grab_real(self):
        if self.index < len(self.filenames):
            image = read_image(self.filenames[self.index])

            y_region = self.roi_y0 + self.roi_height
            x_region = self.roi_x0 + self.roi_width

            result = image[self.roi_y0.magnitude:y_region.magnitude,
                           self.roi_x0.magnitude:x_region.magnitude]
            self.index += 1
        else:
            result = None

        return result


class BufferedCamera(Camera, base.BufferedMixin):

    def _readout_real(self):
        for i in range(3):
            yield self.grab()
