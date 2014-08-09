import numpy as np
from nose.plugins.attrib import attr
from concert.quantities import q
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.tests import slow, TestCase
from concert.tests.util.rotationaxis import SimulationCamera
from concert.processes import scan
from concert.measures import rotation_axis


class TestRotationAxisMeasure(TestCase):

    def setUp(self):
        super(TestRotationAxisMeasure, self).setUp()
        self.x_motor = ContinuousRotationMotor()
        self.y_motor = ContinuousRotationMotor()
        self.z_motor = ContinuousRotationMotor()

        # The bigger the image size, the more images we need to determine
        # the center correctly.
        self.image_source = SimulationCamera(128, self.x_motor["position"],
                                             self.y_motor["position"],
                                             self.z_motor["position"])

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2.0 / self.image_source.rotation_radius) * q.rad

    def make_images(self, x_angle, z_angle, intervals=10):
        self.x_motor.position = z_angle
        self.z_motor.position = x_angle
        result = [f.result()[1] for f in scan(self.y_motor["position"],
                  self.image_source.grab, minimum=0 * q.rad,
                  maximum=2 * np.pi * q.rad,
                  intervals=intervals)]

        return result

    def align_check(self, x_angle, z_angle):
        images = self.make_images(x_angle, z_angle)
        phi, psi = rotation_axis(images)[:2]

        assert phi + x_angle < self.eps
        assert np.abs(psi) - np.abs(z_angle) < self.eps

    def center_check(self, images):
        center = rotation_axis(images)[2]

        assert np.abs(center[1] - self.image_source.ellipse_center[1]) < 2
        assert np.abs(center[0] - self.image_source.ellipse_center[0]) < 2

    @slow
    def test_out_of_fov(self):
        images = np.ones((10, self.image_source.size,
                          self.image_source.size))
        with self.assertRaises(ValueError) as ctx:
            rotation_axis(images)

        self.assertEqual("No sample tip points found.", str(ctx.exception))

    @slow
    def test_center_no_rotation(self):
        images = self.make_images(0 * q.deg, 0 * q.deg, intervals=15)
        self.center_check(images)

    @slow
    def test_center_only_x(self):
        images = self.make_images(17 * q.deg, 0 * q.deg, intervals=15)
        self.center_check(images)

    @slow
    def test_center_only_z(self):
        images = self.make_images(0 * q.deg, 11 * q.deg, intervals=15)
        self.center_check(images)

    @slow
    def test_center_positive(self):
        images = self.make_images(17 * q.deg, 11 * q.deg, intervals=15)
        self.center_check(images)

    @slow
    def test_center_negative_positive(self):
        images = self.make_images(-17 * q.deg, 11 * q.deg, intervals=15)
        self.center_check(images)

    @slow
    def test_center_positive_negative(self):
        images = self.make_images(17 * q.deg, -11 * q.deg, intervals=15)
        self.center_check(images)

    @slow
    def test_center_negative(self):
        images = self.make_images(-17 * q.deg, -11 * q.deg, intervals=15)
        self.center_check(images)

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
