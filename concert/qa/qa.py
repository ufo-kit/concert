"""
qa.py
-----
Encapsulates the quality assurance utilities.
"""
import numpy as np
from concert.base import AsyncObject
from concert.coroutines.base import background

class QualityAssurance(AsyncObject):

    async def __ainit__(self, device) -> None:
        self._device = device
        await self._device.write_attribute("lpf_size", 5)
        await self._device.write_attribute("num_markers", 5)
        await self._device.write_attribute("rot_angle", np.pi)
        await self._device.write_attribute("num_proj", 3000)
        await self._device.write_attribute("wait_interval", 75)
        await self._device.prepare_angular_distribution()
        super().__ainit__()

    @background
    async def derive_rot_axis(self) -> None:
        await self._device.derive_rot_axis()


if __name__ == "__main__":
    pass

