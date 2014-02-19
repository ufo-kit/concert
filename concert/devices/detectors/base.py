"""
Detector is a device which consists of a camera and an objective lens
which affects the effective pixel size.
"""
from concert.quantities import q
from concert.base import Quantity
from concert.devices.base import Device


class Detector(Device):
    """
    Detector consisting of a :class:`.concert.devices.cameras.base.Camera`
    and a magnification given by an objective lens.
    """

    pixel_width = Quantity(q.m, help="Effective pixel width")
    pixel_height = Quantity(q.m, help="Effective pixel height")

    def __init__(self, camera, magnification):
        super(Detector, self).__init__()
        self.camera = camera
        self.magnification = magnification

    def _get_pixel_width(self):
        return self.camera.sensor_pixel_width * self.magnification

    def _get_pixel_height(self):
        return self.camera.sensor_pixel_height * self.magnification
