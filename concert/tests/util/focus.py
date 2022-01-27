import scipy.misc
from scipy.ndimage import fourier
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
        return fourier.fourier_gaussian(self._original, sigma)
