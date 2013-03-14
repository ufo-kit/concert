from concert.base import launch
from concert.devices.base import Device


class Camera(Device):
    """Base class for remotely controllable cameras."""

    def __init__(self):
        super(Camera, self).__init__()

    def record(self, blocking=False):
        """Start recording frames."""
        launch(self._record_real, blocking=blocking)

    def stop(self, blocking=False):
        """Stop recording frames."""
        launch(self._stop_real, blocking=blocking)

    def _record_real(self):
        raise NotImplemented

    def _stop_real(self):
        raise NotImplemented
