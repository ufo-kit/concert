"""
Detector is a device which consists of a camera and an objective lens
which affects the effective pixel size.
"""
from abc import abstractmethod

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

    async def _get_pixel_width(self):
        return (await self.camera.get_sensor_pixel_width()) * self.magnification

    async def _get_pixel_height(self):
        return (await self.camera.get_sensor_pixel_height()) * self.magnification

    @property
    @abstractmethod
    def camera(self):
        ...

    @property
    @abstractmethod
    def magnification(self):
        ...
