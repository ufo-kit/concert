import scipy.misc
from scipy.ndimage import gaussian_filter
from concert.quantities import q
from concert.devices.cameras.dummy import Base as DummyCameraBase


FOCUS_POSITION = 35 * q.mm


class BlurringCamera(DummyCameraBase):

    def __init__(self, motor):
        super(BlurringCamera, self).__init__()
        self._original = scipy.misc.ascent()
        self.motor = motor

    async def _grab_real(self):
        sigma = abs((await self.motor.get_position() - FOCUS_POSITION).magnitude)
        return gaussian_filter(self._original, sigma)
