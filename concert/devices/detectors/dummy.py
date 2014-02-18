"""A dummy detector."""
from concert.devices.detectors import base
from concert.devices.cameras.dummy import Camera


class Detector(base.Detector):

    """A dummy detector."""

    def __init__(self, camera=None, magnification=None):
        camera = Camera() if camera is None else camera
        magnification = 3 if magnification is None else magnification
        super(Detector, self).__init__(camera, magnification)
