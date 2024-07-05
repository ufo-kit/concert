"""
qa.py
-----
Encapsulates the quality assurance utilities.
"""
import numpy as np
from concert.base import AsyncObject
from concert.coroutines.base import background

class QualityAssurance(AsyncObject):

    async def __ainit__(self, device, num_darks: int, num_flats: int,
                        num_radios: int, num_markers: int, rot_angle: np.float64,
                        wait_window: int, lpf_size: int = 5, estm_offset: int = 5,
                        monitor_window: int = 200) -> None:
        self._device = device
        await self._device.write_attribute("lpf_size", lpf_size)
        await self._device.write_attribute("num_markers", num_markers)
        await self._device.write_attribute("rot_angle", rot_angle)
        await self._device.write_attribute("num_darks", num_darks)
        await self._device.write_attribute("num_flats", num_flats)
        await self._device.write_attribute("num_radios", num_radios)
        await self._device.write_attribute("wait_window", wait_window)
        await self._device.write_attribute("estm_offset", estm_offset)
        await self._device.write_attribute("monitor_window", monitor_window)
        await self._device.prepare_angular_distribution()
        await super().__ainit__()

    @background
    async def derive_rot_axis(self) -> None:
        await self._device.derive_rot_axis()


if __name__ == "__main__":
    pass

