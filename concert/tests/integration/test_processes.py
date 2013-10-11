import scipy
from scipy.ndimage import fourier
from concert.tests import assert_almost_equal
from concert.quantities import q
from concert.processes import focus
from concert.devices.cameras.base import Camera
from concert.devices.motors.dummy import Motor
from concert.tests.base import suppressed_logging


MIN_POSITION = 0 * q.mm
MAX_POSITION = 100 * q.mm
FOCUS_POSITION = 35 * q.mm


class BlurringCamera(Camera):

    def __init__(self, motor):
        self._original = scipy.misc.lena()
        self.motor = motor

    def _grab_real(self):
        sigma = abs((self.motor.position - FOCUS_POSITION).magnitude)
        return fourier.fourier_gaussian(self._original, sigma)

    def _record_real(self):
        pass

    def _stop_real(self):
        pass

    def _trigger_real(self):
        pass


@suppressed_logging
def test_focusing():
    motor = Motor(hard_limits=(MIN_POSITION.magnitude,
                               MAX_POSITION.magnitude))
    motor.position = 85. * q.mm
    camera = BlurringCamera(motor)
    focus(camera, motor).wait()
    assert_almost_equal(motor.position, FOCUS_POSITION, 1e-2)
