"""
A :class:`Camera` can be set via the device-specific properties that can be set
and read with :meth:`.Parameter.set` and :meth:`.Parameter.get`.  Moreover, a
camera provides means to

* :meth:`~Camera.start_recording` frames,
* :meth:`~Camera.stop_recording` the acquisition,
* :meth:`~Camera.trigger` a frame capture and
* :meth:`~Camera.grab` to get the last frame.

To setup and use a camera in a typical environment, you would do::

    import numpy as np
    from concert.devices.cameras.uca import UcaCamera

    camera = UcaCamera('pco')
    camera.exposure_time = 0.2 * q.s
    camera.start_recording().wait()
    camera.trigger().wait()
    data = camera.grab()
    camera.stop_recording()

    print("mean=%f, stddev=%f" % (np.mean(data), np.std(data))
"""
from concert.quantities import q
from concert.devices.base import Device, Parameter


class CameraError(Exception):

    """Camera specific errors."""
    pass


class Camera(Device):

    """Base class for remotely controllable cameras.

    .. py:attribute:: frame-rate

        Frame rate of acquisition in q.count per time unit.
    """

    def __init__(self, params=None):
        frame_rate_param = Parameter(name='frame-rate',
                                     fget=self._get_frame_rate,
                                     fset=self._set_frame_rate,
                                     unit=q.count / q.second,
                                     doc="Frame rate of image acquisition")

        if params is not None:
            params.append(frame_rate_param)
        else:
            params = [frame_rate_param]

        super(Camera, self).__init__(params)

    def start_recording(self):
        """Start recording frames."""
        self._record_real()

    def stop_recording(self):
        """Stop recording frames."""
        self._stop_real()

    def trigger(self):
        """Trigger a frame if possible."""
        self._trigger_real()

    def grab(self):
        """Return a NumPy array with data of the current frame."""
        return self._grab_real()

    def _get_frame_rate(self):
        raise NotImplementedError

    def _set_frame_rate(self, frame_rate):
        raise NotImplementedError

    def _record_real(self):
        raise NotImplementedError

    def _stop_real(self):
        raise NotImplementedError

    def _trigger_real(self):
        raise NotImplementedError

    def _grab_real(self):
        raise NotImplementedError
