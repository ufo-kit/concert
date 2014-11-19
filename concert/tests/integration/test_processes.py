import scipy
from scipy.ndimage import fourier
from concert.tests import assert_almost_equal, TestCase
from concert.quantities import q
from concert.processes import focus
from concert.devices.cameras.dummy import Base as DummyCameraBase
from concert.devices.motors.dummy import LinearMotor


MIN_POSITION = 0 * q.mm
MAX_POSITION = 100 * q.mm
FOCUS_POSITION = 35 * q.mm


class BlurringCamera(DummyCameraBase):

    def __init__(self, motor):
        super(BlurringCamera, self).__init__()
        self._original = scipy.misc.lena()
        self.motor = motor

    def _grab_real(self):
        sigma = abs((self.motor.position - FOCUS_POSITION).magnitude)
        return fourier.fourier_gaussian(self._original, sigma)


class TestProcesses(TestCase):

    def setUp(self):
        self.motor = LinearMotor()

    def test_focusing(self):
        self.motor.position = 40. * q.mm
        camera = BlurringCamera(self.motor)
        focus(camera, self.motor).join()
        assert_almost_equal(self.motor.position, FOCUS_POSITION, 1e-2)
