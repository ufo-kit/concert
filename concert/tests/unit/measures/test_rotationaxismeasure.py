import unittest
import logbook
import numpy as np
from testfixtures import ShouldRaise
from concert.quantities import q
from concert.measures.rotationaxis import Ellipse
from concert.tests.util.rotationaxis import SimulationCamera
from concert.devices.base import LinearCalibration
from concert.devices.motors.dummy import Motor
from concert.tests import slow
from concert.processes.scan import Scanner


class TestRotationAxisMeasure(unittest.TestCase):

    def setUp(self):
        self.handler = logbook.TestHandler()
        self.handler.push_application()
        self.x_motor = Motor(LinearCalibration(q.count / q.deg, 0 * q.deg),
                             hard_limits=(-1e5, 1e5))
        self.x_motor["position"].unit = q.deg
        self.y_motor = Motor(LinearCalibration(q.count / q.deg, 0 * q.deg),
                             hard_limits=(-1e5, 1e5))
        self.y_motor["position"].unit = q.deg
        self.z_motor = Motor(LinearCalibration(q.count / q.deg, 0 * q.deg),
                             hard_limits=(-1e5, 1e5))
        self.z_motor["position"].unit = q.deg

        # The bigger the image size, the more images we need to determine
        # the center correctly.
        self.image_source = SimulationCamera(256, self.x_motor["position"],
                                             self.y_motor["position"],
                                             self.z_motor["position"])

        # A scanner which scans the rotation axis.
        self.scanner = Scanner(
            self.y_motor["position"], self.image_source.grab)
        self.scanner.minimum = 0 * q.rad
        self.scanner.maximum = 2 * np.pi * q.rad
        self.scanner.intervals = 50

        self.measure = Ellipse()

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2.0 / self.image_source.rotation_radius) * q.rad

    def tearDown(self):
        self.handler.pop_application()

    def make_images(self, x_angle, z_angle):
        self.x_motor.position = z_angle
        self.z_motor.position = x_angle
        self.measure.images = self.scanner.run().result()[1]

    def align_check(self, x_angle, z_angle):
        self.make_images(x_angle, z_angle)
        phi, psi = self.measure()

        assert phi + x_angle < self.eps
        assert np.abs(psi) - np.abs(z_angle) < self.eps

    def center_check(self):
        assert np.abs(self.measure.center[1] -
                      self.image_source.ellipse_center[1]) < 1
        assert np.abs(self.measure.center[0] -
                      self.image_source.ellipse_center[0]) < 1

    @slow
    def test_out_of_fov(self):
        with ShouldRaise(ValueError("No sample tip points found.")):
            self.measure.images = np.ones((self.scanner.intervals,
                                           self.image_source.size,
                                           self.image_source.size))
            self.measure()

    @slow
    def test_center_no_rotation(self):
        self.make_images(0 * q.deg, 0 * q.deg)
        self.center_check()

    @slow
    def test_center_only_x(self):
        self.make_images(17 * q.deg, 0 * q.deg)
        self.center_check()

    @slow
    def test_center_only_z(self):
        self.make_images(0 * q.deg, 11 * q.deg)
        self.center_check()

    @slow
    def test_center_positive(self):
        self.make_images(17 * q.deg, 11 * q.deg)
        self.center_check()

    @slow
    def test_center_negative_positive(self):
        self.make_images(-17 * q.deg, 11 * q.deg)
        self.center_check()

    @slow
    def test_center_positive_negative(self):
        self.make_images(17 * q.deg, -11 * q.deg)
        self.center_check()

    @slow
    def test_center_negative(self):
        self.make_images(-17 * q.deg, -11 * q.deg)
        self.center_check()

    @slow
    def test_only_x(self):
        """Only misaligned laterally."""
        self.align_check(0 * q.deg, 0 * q.deg)

    @slow
    def test_only_z(self):
        """Only misaligned in the beam direction."""
        self.align_check(0 * q.deg, 11 * q.deg)

    @slow
    def test_huge_x(self):
        self.image_source.scale = (3, 0.25, 3)
        self.align_check(60 * q.deg, 11 * q.deg)

    @slow
    def test_huge_z(self):
        self.image_source.scale = (3, 0.25, 3)
        self.align_check(11 * q.deg, 60 * q.deg)

    @slow
    def test_positive(self):
        self.align_check(17 * q.deg, 11 * q.deg)

    @slow
    def test_negative_positive(self):
        self.align_check(-17 * q.deg, 11 * q.deg)

    @slow
    def test_positive_negative(self):
        self.align_check(17 * q.deg, -11 * q.deg)

    @slow
    def test_negative(self):
        self.align_check(-17 * q.deg, -11 * q.deg)
