import numpy as np
from concert.quantities import q
from concert.devices.motors.dummy import ContinuousRotationMotor
from concert.imageprocessing import find_needle_tips, find_sphere_centers
from concert.processes.common import align_rotation_axis
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

    def align_check(self, x_angle, z_angle, has_x_motor=True, has_z_motor=True,
                    get_ellipse_points=find_needle_tips, get_ellipse_points_kwargs=None):
        """"Align and check the results."""
        self.x_motor.position = z_angle
        self.z_motor.position = x_angle

        x_motor = self.x_motor if has_x_motor else None
        z_motor = self.z_motor if has_z_motor else None

        align_rotation_axis(self.camera, self.y_motor, x_motor=x_motor, z_motor=z_motor,
                            get_ellipse_points=get_ellipse_points,
                            get_ellipse_points_kwargs=get_ellipse_points_kwargs).join()

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
    def test_sphere(self):
        """Test sphere instead of needle."""
        self.camera.rotation_radius = self.camera.size / 2
        self.camera.scale = (.5, .5, .5)
        self.camera.y_position = 0
        self.eps = np.arctan(2.0 / self.camera.rotation_radius) * q.rad
        self.align_check(-17 * q.deg, 11 * q.deg,
                         get_ellipse_points=find_sphere_centers)

    @slow
    def test_kwargs(self):
        """Test keyword arguments passing. Use sphere simulation and segmentation for that."""
        self.camera.rotation_radius = self.camera.size / 2
        self.camera.scale = (.25, .25, .25)
        self.camera.y_position = 0
        self.eps = np.arctan(2.0 / self.camera.rotation_radius) * q.rad
        self.align_check(-17 * q.deg, 11 * q.deg,
                         get_ellipse_points=find_sphere_centers,
                         get_ellipse_points_kwargs={'correlation_threshold': 0.9})
