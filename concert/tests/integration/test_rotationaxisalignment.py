import numpy as np
from nose.plugins.attrib import attr
from concert.quantities import q
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.processes import align_rotation_axis
from concert.tests import slow, TestCase
from concert.tests.util.rotationaxis import SimulationCamera


class TestDummyAlignment(TestCase):

    def setUp(self):
        super(TestDummyAlignment, self).setUp()
        self.x_motor = ContinuousRotationMotor()
        self.y_motor = ContinuousRotationMotor()
        self.z_motor = ContinuousRotationMotor()

        self.x_motor.position = 0 * q.deg
        self.y_motor.position = 0 * q.deg
        self.z_motor.position = 0 * q.deg

        self.camera = SimulationCamera(128, self.x_motor["position"],
                                       self.y_motor["position"],
                                       self.z_motor["position"])

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2.0 / self.camera.rotation_radius) * q.rad

    def align_check(self, x_angle, z_angle, has_x_motor=True,
                    has_z_motor=True, flat=None, dark=None):
        """"Align and check the results."""
        self.x_motor.position = z_angle
        self.z_motor.position = x_angle

        x_motor = self.x_motor if has_x_motor else None
        z_motor = self.z_motor if has_z_motor else None

        align_rotation_axis(self.camera, self.y_motor, x_motor=x_motor, z_motor=z_motor, flat=flat,
                dark=dark).join()

        # In our case the best perfectly aligned position is when both
        # motors are in 0.
        if has_x_motor:
            assert np.abs(self.x_motor.position) < self.eps
        if has_z_motor:
            assert np.abs(self.z_motor.position) < self.eps

    def test_no_motor(self):
        with self.assertRaises(ValueError):
            align_rotation_axis(self.camera, self.y_motor, x_motor=None, z_motor=None).join()

    @slow
    def test_out_of_fov(self):
        def get_ones():
            return np.ones((self.camera.size,
                            self.camera.size))

        self.camera._grab_real = get_ones

        with self.assertRaises(ValueError) as ctx:
            align_rotation_axis(self.camera, self.y_motor,
                                x_motor=self.x_motor,
                                z_motor=self.z_motor).join()

        self.assertEqual("No sample tip points found.", str(ctx.exception))

    @slow
    def test_not_offcentered(self):
        self.camera.rotation_radius = 0
        with self.assertRaises(ValueError) as ctx:
            align_rotation_axis(self.camera, self.y_motor,
                                x_motor=self.x_motor,
                                z_motor=self.z_motor).join()

        self.assertEqual("Sample off-centering too " +
                         "small, enlarge rotation radius.",
                         str(ctx.exception))

    @slow
    def test_no_x_axis(self):
        """Test the case when there is no x-axis motor available."""
        self.align_check(17 * q.deg, 11 * q.deg, has_z_motor=False)

    @slow
    def test_no_z_axis(self):
        """Test the case when there is no x-axis motor available."""
        self.align_check(17 * q.deg, 11 * q.deg, has_x_motor=False)

    @slow
    def test_not_misaligned(self):
        "Perfectly aligned rotation axis."
        self.align_check(0 * q.deg, 0 * q.deg)

    @slow
    def test_only_x(self):
        """Only misaligned laterally."""
        self.align_check(-17 * q.deg, 0 * q.deg)

    @slow
    def test_only_z(self):
        """Only misaligned in the beam direction."""
        self.align_check(0 * q.deg, 11 * q.deg)

    @slow
    def test_huge_x(self):
        self.camera.scale = (3, 0.25, 3)
        self.align_check(60 * q.deg, 11 * q.deg)

    @slow
    def test_huge_z(self):
        self.camera.scale = (3, 0.25, 3)
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

    @slow
    def test_flat_corrected(self):
        shape = (self.camera.size, self.camera.size)
        dark = np.zeros(shape, dtype=np.float)
        flat = np.ones(shape, dtype=np.float)
        self.align_check(-17 * q.deg, -11 * q.deg, flat=flat, dark=dark)
