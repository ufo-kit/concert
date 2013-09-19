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
        params = [Parameter('fps', unit=q.count / q.s),
                  Parameter('trigger_mode', lower=FileCamera.TRIGGER_AUTO,
                  upper=FileCamera.TRIGGER_SOFTWARE)]
        super(FileCamera, self).__init__(params)

        self._recording = None
        self._stopped = False
        self._index = -1
        self._start_time = None
        self._stop_time = None
        self._trigger = Event()
        self._trigger_time = None
        self._files = [os.path.join(folder, file_name) for file_name in
                       sorted(os.listdir(folder))]

    def _get_index(self, stop_time=None):
        if stop_time is None:
            stop_time = time.time() * q.s

        return int(self.fps * (stop_time - self._start_time))

    def _record_real(self):
        self._start_time = time.time() * q.s
        self._recording = True

    def _stop_real(self):
        if not self._recording:
            raise CameraError("start_recording() not called")
        self._stop_time = time.time() * q.s
        self._recording = False
        self._stopped = True

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
                time.sleep(1.0 / self.fps)
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

        return read_image(self._files[self._index]).result()


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
        cur_time = self.exposure_time.to(q.s).magnitude

        # 1e5 is a dummy correlation between exposure time and emitted e-.
        tmp = self._background + cur_time * 1e5
        max_value = np.iinfo(np.uint16).max
        tmp = np.random.poisson(tmp)
        # Cut values beyond the bit-depth.
        tmp[tmp > max_value] = max_value

        return np.cast[np.uint16](tmp)
