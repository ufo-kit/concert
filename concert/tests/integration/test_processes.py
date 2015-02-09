import numpy as np
import scipy
from scipy.ndimage import fourier
from concert.tests import assert_almost_equal, TestCase
from concert.quantities import q
from concert.devices.cameras.dummy import Base as DummyCameraBase, Camera
from concert.devices.motors.dummy import LinearMotor, RotationMotor
from concert.devices.shutters.dummy import Shutter
from concert.processes.common import focus
from concert.processes.beamline import (acquire_dark, acquire_image_with_beam,
                                        determine_rotation_axis)


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
        self.camera = Camera()
        self.shutter = Shutter()

    def test_focusing(self):
        self.motor.position = 40. * q.mm
        camera = BlurringCamera(self.motor)
        focus(camera, self.motor).join()
        assert_almost_equal(self.motor.position, FOCUS_POSITION, 1e-2)

    def test_acquire_dark(self):
        self.assertTrue(isinstance(acquire_dark(self.camera, self.shutter).result(), np.ndarray))

    def test_acquire_image_with_beam(self):
        frame = acquire_image_with_beam(self.camera, self.shutter, self.motor, 1 * q.mm).result()
        self.assertTrue(isinstance(frame, np.ndarray))
        self.assertEqual(self.motor.position, 1 * q.mm)

    def test_determine_rotation_axis(self):
        rot_motor = RotationMotor()
        axis = determine_rotation_axis(self.camera, self.shutter, self.motor, rot_motor,
                                       1 * q.mm, 3 * q.mm).result()
        self.assertTrue(isinstance(axis, q.Quantity))
