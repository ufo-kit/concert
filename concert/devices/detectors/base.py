"""
Detector is a device which consists of a camera and an objective lens
which affects the effective pixel size.
"""
from concert.quantities import q
from concert.base import Quantity
from concert.devices.base import Device


class Detector(Device):
    """
    A base detector class for holding cameras and optics necessary to
    do acquire image.
    """

    pixel_width = Quantity(q.m, help="Effective pixel width")
    pixel_height = Quantity(q.m, help="Effective pixel height")

    def __init__(self):
        super(Detector, self).__init__()
        self._camera = None
        self._magnification = None

    def _get_pixel_width(self):
        return self.camera.sensor_pixel_width * self.magnification

    def _get_pixel_height(self):
        return self.camera.sensor_pixel_height * self.magnification

    @property
    def camera(self):
        return self._camera

    @property
    def magnification(self):
        return self._magnification
