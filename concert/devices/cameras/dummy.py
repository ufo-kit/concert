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
        self._trigger_source = self.trigger_sources.AUTO
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

    def _get_trigger_source(self):
        return self._trigger_source

    def _set_trigger_source(self, source):
        self._trigger_source = source

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

    def __init__(self, background=None, simulate=True):
        """
        *background* can be an array-like that will be used to generate the frame when calling
        :meth:`.grab`. If *simulate* is True the final image intensity will be scaled based on
        exposure time and poisson noise will be added. If *simulate* is False, the background will
        be returned with no modifications to it.
        """
        super(Camera, self).__init__()
        self.simulate = simulate

        if background is not None:
            self.roi_width = background.shape[1] * q.pixel
            self.roi_height = background.shape[0] * q.pixel
            self._background = background
        else:
            shape = (640, 480)
            self.roi_width, self.roi_height = shape * q.pixel
            self._background = np.ones(shape[::-1], dtype=np.uint8)

    def _grab_real(self):
        if not self.simulate:
            return self._background
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
    from the beginning when the recording starts. *start_index* specifies the index of the first
    read image (not file index, in case the files are multi-page).
    """

    def __init__(self, directory, reset_on_start=True, start_index=0):
        # Let users change the directory
        self.directory = directory
        super(FileCamera, self).__init__()

        self._start_index = start_index
        self.index = 0
        self._image_index = 0
        self.reset_on_start = reset_on_start
        self.filenames = [os.path.join(directory, file_name) for file_name in
                          sorted(os.listdir(directory))]

        if not self.filenames:
            raise base.CameraError("No files found")

        self._read_next_file()
        self._fastforward()
        self._roi_width = self._image.shape[-1] * q.pixel
        self._roi_height = self._image.shape[-2] * q.pixel

    @transition(target='recording')
    def _record_real(self):
        if self.reset_on_start:
            self.index = 0
            self._image_index = 0
            self._read_next_file()
            self._fastforward()

    def _read_next_file(self):
        if self.index < len(self.filenames):
            self._image = read_image(self.filenames[self.index])
            self._image_index = 0
            self.index += 1
            if self._image.ndim == 2:
                # Make sure the image is 3D so that grab can be simpler
                self._image = self._image[np.newaxis, :]
        else:
            self._image = None

    def _fastforward(self):
        image_index = self._image.shape[0]
        while image_index <= self._start_index:
            self._read_next_file()
            if self._image is None:
                raise base.CameraError('Not enough files for specified start index')
            image_index += self._image.shape[0]

        self._image_index = self._start_index - image_index + self._image.shape[0]

    def _grab_real(self):
        if self._image is None:
            result = None
        else:
            y_region = self.roi_y0 + self.roi_height
            x_region = self.roi_x0 + self.roi_width
            result = self._image[self._image_index,
                                 self.roi_y0.magnitude:y_region.magnitude,
                                 self.roi_x0.magnitude:x_region.magnitude]
            self._image_index += 1
            if self._image_index == self._image.shape[0]:
                self._read_next_file()

        return result


class BufferedCamera(Camera, base.BufferedMixin):

    def _readout_real(self):
        for i in range(3):
            yield self.grab()
