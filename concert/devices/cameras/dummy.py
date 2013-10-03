"""This module provides a simple dummy camera."""
import numpy as np
from concert.quantities import q
from concert.base import Parameter
from concert.devices.cameras import base
from threading import Event
import os
import time
from concert.storage import read_image
from concert.devices.cameras.base import CameraError


class FileCamera(base.Camera):

    """A camera that reads files in a *folder*."""
    TRIGGER_AUTO = 0
    TRIGGER_SOFTWARE = 1

    def __init__(self, folder):
        # Let users change the folder
        self.folder = folder
        params = [Parameter('trigger-mode', lower=FileCamera.TRIGGER_AUTO,
                  upper=FileCamera.TRIGGER_SOFTWARE)]
        super(FileCamera, self).__init__(params)

        self._frame_rate = None
        self._recording = None
        self._index = -1
        self._start_time = None
        self._stop_time = None
        self._trigger = Event()
        self._trigger_time = None
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = None
        self.roi_height = None
        self._files = [os.path.join(folder, file_name) for file_name in
                       sorted(os.listdir(folder))]

    def _get_index(self, stop_time=None):
        if stop_time is None:
            stop_time = time.time() * q.s

        return int(self.frame_rate * (stop_time - self._start_time))

    def _get_frame_rate(self):
        return self._frame_rate

    def _set_frame_rate(self, frame_rate):
        self._frame_rate = frame_rate

    def _record_real(self):
        self._start_time = time.time() * q.s
        self._recording = True

    def _stop_real(self):
        if not self._recording:
            raise CameraError("start_recording() not called")
        self._stop_time = time.time() * q.s
        self._trigger.clear()
        self._trigger_time = None
        self._recording = False

    def _trigger_real(self):
        if self.trigger_mode == FileCamera.TRIGGER_SOFTWARE:
            self._trigger.set()
            self._trigger_time = time.time()
        else:
            raise CameraError("Cannot trigger in current trigger mode")

    def _grab_real(self):
        if self._recording is None:
            raise CameraError("Camera hasn't been in recording mode yet")

        if self._recording:
            if self.trigger_mode == FileCamera.TRIGGER_SOFTWARE:
                self._trigger.wait()
                self._trigger.clear()

            if self._get_index() == self._index:
                # Wait for the next frame if the readout is too fast
                time.sleep(1.0 / self.frame_rate)
            self._index = self._get_index()
        else:
            if self.trigger_mode == FileCamera.TRIGGER_AUTO:
                self._index += 1
            else:
                if self._trigger.is_set():
                    self._trigger.clear()
                else:
                    # Camera hasn't been triggered, don't return anything
                    return None
                self._index = self._get_index(self._trigger_time)

            if self._index > self._get_index(self._stop_time) or \
                    self._index >= len(self._files):
                return None

        image = read_image(self._files[self._index]).result()
        if self.roi_height is None:
            y_region = image.shape[0]
        else:
            y_region = self.roi_y + self.roi_height

        if self.roi_width is None:
            x_region = image.shape[1]
        else:
            x_region = self.roi_x + self.roi_width

        return image[self.roi_y:y_region, self.roi_x:x_region]


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

        self._frame_rate = None
        self.exposure_time = 1 * q.ms
        self.sensor_pixel_width = 5 * q.micrometer
        self.sensor_pixel_height = 5 * q.micrometer

        if background is not None:
            self.roi_width = background.shape[1]
            self.roi_height = background.shape[0]
            self._background = background
        else:
            shape = (640, 480)
            self.roi_width, self.roi_height = shape
            self._background = np.ones(shape)

    def _get_frame_rate(self):
        return self._frame_rate

    def _set_frame_rate(self, frame_rate):
        self._frame_rate = frame_rate

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
        to_sleep = q.count / self.frame_rate
        to_sleep = to_sleep.to_base_units() - duration * q.s
        if to_sleep > 0 * q.s:
            time.sleep(to_sleep.to_base_units() - duration * q.s)

        return np.cast[np.uint16](tmp)
