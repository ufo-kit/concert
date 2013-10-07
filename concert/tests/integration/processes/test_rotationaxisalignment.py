import numpy as np
from nose.plugins.attrib import attr
from concert.quantities import q
from concert.devices.motors.dummy import Motor
from concert.devices.base import LinearCalibration
from concert.processes import scan, align_rotation_axis
from concert.tests import slow
from concert.tests.util.rotationaxis import SimulationCamera
from concert.tests.base import ConcertTest
from concert.measures import get_rotation_axis


@attr('skip-travis')
class TestDummyAlignment(ConcertTest):

    def setUp(self):
        super(TestDummyAlignment, self).setUp()
        calibration = LinearCalibration(q.count / q.deg, 0 * q.deg)
        hard_limits = (-np.Inf * q.count, np.Inf * q.count)
        self.x_motor = Motor(calibration=calibration, hard_limits=hard_limits)
        self.y_motor = Motor(calibration=calibration, hard_limits=hard_limits)
        self.z_motor = Motor(calibration=calibration, hard_limits=hard_limits)

        self.x_motor.position = 0 * q.deg
        self.z_motor.position = 0 * q.deg

        self.image_source = SimulationCamera(128, self.x_motor["position"],
                                             self.y_motor["position"],
                                             self.z_motor["position"])

        self.feedback = self.image_source.grab

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2.0 / self.image_source.rotation_radius) * q.rad

    def get_images(self):
        return scan(self.y_motor["position"], self.feedback, minimum=0 * q.rad,
                    maximum=2 * np.pi * q.rad, intervals=10).result()[1]

    def align_check(self, x_angle, z_angle, has_z_motor=True):
        """"Align and check the results."""
        self.x_motor.position = z_angle
        self.z_motor.position = x_angle

        z_motor = self.z_motor if has_z_motor else None

        align_rotation_axis(get_rotation_axis, self.get_images,
                            self.x_motor, z_motor).wait()

        # In our case the best perfectly aligned position is when both
        # motors are in 0.
        assert np.abs(self.x_motor.position) < self.eps
        if has_z_motor:
            assert np.abs(self.z_motor.position) < self.eps

    @slow
    def test_out_of_fov(self):
        def get_ones():
            return np.ones((self.image_source.size,
                            self.image_source.size))

        self.feedback = get_ones
        with self.assertRaises(ValueError) as ctx:
            align_rotation_axis(get_rotation_axis, self.get_images,
                                self.x_motor, self.z_motor).wait()

        self.assertEqual("No sample tip points found.", ctx.exception.message)

    @slow
    def test_not_offcentered(self):
        self.image_source.rotation_radius = 0
        with self.assertRaises(ValueError) as ctx:
            align_rotation_axis(get_rotation_axis, self.get_images,
                                self.x_motor, self.z_motor).wait()

        self.assertEqual("Sample off-centering too " +
                         "small, enlarge rotation radius.",
                         ctx.exception.message)

    @slow
    def test_no_x_axis(self):
        """Test the case when there is no x-axis motor available."""
        self.align_check(17 * q.deg, 11 * q.deg, has_z_motor=False)

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
