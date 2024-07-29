"""
qa.py
-----
Encapsulates quality assurance stack for data acquisition and online reconstruction.
#TODO: Quality Assurance in our context is a broad terminology, because we are also working on
automatic sample alignment and optimization of additional parameters for reconstruction. Therefore,
this module is subjected to undergo severe changes.
"""
import numpy as np
from concert.base import AsyncObject
from concert.coroutines.base import background

class QualityAssurance(AsyncObject):

    async def __ainit__(self, device, num_darks: int, num_flats: int, num_radios: int,
                        rot_angle: float, estm_offset: int = 5) -> None:
        """
        Encapsulates quality assurance stack for data acquisition and online reconstruction.
        #TODO: Quality Assurance in our context is a broad terminology, because we are also working on
        automatic sample alignment and optimization of additional parameters for reconstruction. Therefore,
        this module is subjected to undergo severe changes.

        :param device: tango device proxy for the quality assurance device server
        :type device: tango.DeviceProxy
        :param num_darks: number of dark fields acquired
        :type num_darks: int
        :param num_flats: number of flat fields acquired
        :type num_flats: int
        :param num_radios: number of projections acquired
        :type num_radios: int
        :param rot_angle: overall angle of rotation in radians
        :type rot_angle: float
        :param estm_offset: offset to use while estimating center of rotation
        :type estm_offset: int
        """
        self._device = device
        await self._device.write_attribute("num_darks", num_darks)
        await self._device.write_attribute("num_flats", num_flats)
        await self._device.write_attribute("num_radios", num_radios)
        await self._device.write_attribute("rot_angle", rot_angle)
        await self._device.write_attribute("estm_offset", estm_offset)
        await self._device.prepare_angular_distribution()
        await super().__ainit__()

    @background
    async def estimate_center_of_rotation(self, num_markers: int, wait_window: int,
                                          sigma: float) -> None:
        await self._device.estimate_center_of_rotation((num_markers, wait_window, sigma))


if __name__ == "__main__":
    pass

