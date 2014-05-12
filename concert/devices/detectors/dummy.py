"""A dummy detector."""
from concert.devices.detectors import base
from concert.devices.cameras.dummy import Camera


class Detector(base.Detector):

    """A dummy detector."""

    def __init__(self, camera=None, magnification=None):
        self._camera = Camera() if camera is None else camera
        self._magnification = 3 if magnification is None else magnification
        super(Detector, self).__init__()

    @property
    def camera(self):
        return self._camera

    @property
    def magnification(self):
        return self._magnification
