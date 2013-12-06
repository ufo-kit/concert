"""
A :class:`Camera` can be set via the device-specific properties that can be set
and read with :meth:`.Parameter.set` and :meth:`.Parameter.get`.  Moreover, a
camera provides means to

* :meth:`~Camera.start_recording` frames,
* :meth:`~Camera.stop_recording` the acquisition,
* :meth:`~Camera.trigger` a frame capture and
* :meth:`~Camera.grab` to get the last frame.

Camera triggering is specified by the :attr:`~Camera.trigger_mode` parameter, which
can be one of

* :attr:`camera.trigger_modes.AUTO` means the camera triggers itself
  automatically, the frames start being recorded right after the
  :meth:`~camera.start_recording` call and stop being recorded by
  :meth:`~camera.stop_recording`

* :attr:`Camera.trigger_modes.SOFTWARE` means the camera needs to be triggered
  by the user by :meth:`~Camera.trigger()`. This way you have complete programatic
  control over when is the camera triggered, example usage::

    camera.trigger_mode = camera.trigger_modes.SOFTWARE
    camera.start_recording()
    camera.trigger()
    long_operation()
    # Here we get the frame from before the long operation
    camera.grab()

* :attr:`Camera.trigger_modes.EXTERNAL` is a mode when the camera is triggered
  by an external low-level signal (such as TTL). This mode provides very precise
  triggering in terms of time synchronization with other devices


To setup and use a camera in a typical environment, you would do::

    import numpy as np
    from concert.devices.cameras.uca import UcaCamera

    camera = UcaCamera('pco')
    camera.trigger_mode = camera.trigger_modes.SOFTWARE
    camera.exposure_time = 0.2 * q.s
    camera.start_recording()
    camera.trigger()
    data = camera.grab()
    camera.stop_recording()

    print("mean=%f, stddev=%f" % (np.mean(data), np.std(data))
"""
from concert.helpers import Bunch
from concert.quantities import q
from concert.fsm import State, transition
from concert.base import Parameter, Quantity
from concert.devices.base import Device


class CameraError(Exception):

    """Camera specific errors."""
    pass


class Camera(Device):

    """Base class for remotely controllable cameras.

    .. py:attribute:: frame-rate

        Frame rate of acquisition in q.count per time unit.
    """

    trigger_modes = Bunch(['AUTO', 'SOFTWARE', 'EXTERNAL'])
    state = State(default='standby')
    frame_rate = Quantity(unit=1.0 / q.second)
    trigger_mode = Parameter()

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

    def _get_trigger_mode(self):
        raise NotImplementedError

    def _set_trigger_mode(self, mode):
        raise NotImplementedError

    def _record_real(self):
        raise NotImplementedError

    def _stop_real(self):
        raise NotImplementedError

    def _trigger_real(self):
        raise NotImplementedError

    def _grab_real(self):
        raise NotImplementedError
