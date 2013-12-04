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
    camera.start_recording()
    camera.trigger()
    data = camera.grab()
    camera.stop_recording()

    print("mean=%f, stddev=%f" % (np.mean(data), np.std(data))
"""
from concert.quantities import q
from concert.fsm import State, transition
from concert.base import Parameter
from concert.devices.base import Device


class CameraError(Exception):

    """Camera specific errors."""
    pass


class Camera(Device):

    """Base class for remotely controllable cameras.

    .. py:attribute:: frame-rate

        Frame rate of acquisition in q.count per time unit.
    """

    state = State(default='standby')
    frame_rate = Parameter(unit=1.0 / q.second)

    def __init__(self):
        super(Camera, self).__init__()

    @transition(source='standby', target='recording')
    def start_recording(self):
        """Start recording frames."""
        self._record_real()

    @transition(source='recording', target='standby')
    def stop_recording(self):
        """Stop recording frames."""
        self._stop_real()

    def trigger(self):
        """Trigger a frame if possible."""
        self._trigger_real()

    def grab(self):
        """Return a NumPy array with data of the current frame."""
        return self._grab_real()

    def acquire(self, num_frames, trigger=False):
        """Acquire *num_frames* frames and *trigger* if necessary."""
        for i in range(num_frames):
            if trigger:
                self.trigger()

            yield self.grab()

    def _record_real(self):
        raise NotImplementedError

    def _stop_real(self):
        raise NotImplementedError

    def _trigger_real(self):
        raise NotImplementedError

    def _grab_real(self):
        raise NotImplementedError
