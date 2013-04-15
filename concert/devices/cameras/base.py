"""
A :class:`Camera` can be set via the device-specific properties that can be set
and read with :meth:`.ConcertObject.set` and :meth:`.ConcertObject.get`.
Moreover, a camera provides means to

* :meth:`Camera.record` frames,
* :meth:`Camera.stop` the acquisition,
* :meth:`Camera.trigger` a frame capture and
* :meth:`Camera.grab` to get the last frame.

To setup and use a camera in a typical environment, you would do::

    import numpy as np
    from concert.devices.cameras.uca import UcaCamera

    camera = UcaCamera('pco')
    camera.exposure_time = 0.2 * q.s
    camera.record().wait()
    camera.trigger().wait()
    data = camera.grab()
    camera.stop()

    print("mean=%f, stddev=%f" % (np.mean(data), np.std(data))
"""
from concert.devices.base import Device


class Camera(Device):
    """Base class for remotely controllable cameras."""

    def __init__(self, params):
        super(Camera, self).__init__(params)

    def record(self):
        """Start recording frames."""
        self._record_real()

    def stop(self):
        """Stop recording frames."""
        self._stop_real()

    def trigger(self):
        """Trigger a frame if possible."""
        self._trigger_real()

    def grab(self):
        """Return a NumPy array with data of the current frame."""
        return self._grab_real()

    def _record_real(self):
        raise NotImplementedError

    def _stop_real(self):
        raise NotImplementedError

    def _trigger_real(self):
        raise NotImplementedError

    def _grab_real(self):
        raise NotImplementedError
