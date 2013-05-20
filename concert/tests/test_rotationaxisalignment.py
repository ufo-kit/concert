import unittest
import numpy as np
from concert.devices.motors.dummy import Motor
import quantities as q
from concert.devices.motors.base import LinearCalibration
from concert.asynchronous import dispatcher
from testfixtures import ShouldRaise
from concert.tests import slow
from threading import Event
from concert.processes.tomoalignment import Aligner
from concert.measures.rotationaxis import Ellipse
from concert.tests.util.rotationaxis import ImageSource


class TestDummyAlignment(unittest.TestCase):
    def setUp(self):
        self.z_motor = Motor(LinearCalibration(1 / q.deg, 0 * q.deg),
                             hard_limits=(-1e5, 1e5))
        self.z_motor["position"].unit = q.deg
        self.x_motor = Motor(LinearCalibration(1 / q.deg, 0 * q.deg),
                             hard_limits=(-1e5, 1e5))
        self.x_motor["position"].unit = q.deg

        self.x_motor.position = 0*q.deg
        self.z_motor.position = 0*q.deg

        self.image_source = ImageSource(256, self.x_motor["position"],
                                        self.z_motor["position"], 10)
        dispatcher.subscribe(self.image_source, self.image_source.ITER,
                             self.iteration_listener)
        self.aligner = Aligner(Ellipse(), self.image_source,
                               self.x_motor, self.z_motor)
        dispatcher.subscribe(self.aligner, self.aligner.AXIS_ALIGNED,
                             self.alignment_listener)

        self.max_iterations = 10

        # Alignment finishes after the aligner finishes or it iterates
        # too much, in which case the test fails.
        self.alignment_finished = Event()

        # Allow 1 px misalignment in y-direction.
        self.eps = np.arctan(2.0/self.image_source.rotation_radius)*q.rad

    def iteration_listener(self, sender):
        if self.image_source.iteration == self.max_iterations:
            self.alignment_finished.set()

    def alignment_listener(self, aligner):
        self.alignment_finished.set()

    def check_alignment(self):
        self.alignment_finished.wait()

        self.assertGreater(self.max_iterations, self.image_source.iteration,
                           "Maximum iterations exceeded.")

    def align_check(self, x_angle, z_angle, has_z_motor=True):
        self.x_motor.position = z_angle
        self.z_motor.position = x_angle

        self.aligner.z_motor = self.z_motor if has_z_motor else None
        self.aligner.run()

        self.check_alignment()

        # In our case the best perfectly aligned position is when both
        # motors are in 0.
        assert np.abs(self.x_motor.position) < self.eps
        if has_z_motor:
            assert np.abs(self.z_motor.position) < self.eps

    def test_out_of_fov(self):
        class DummySource(object):
            def get_images(self):
                return np.ones((10, 256, 256))

        self.aligner = Aligner(
            Ellipse(), DummySource(), self.x_motor, self.z_motor)
        with ShouldRaise(ValueError("No sample tip points found.")):
            self.aligner.run().wait()

    @slow
    def test_not_offcentered(self):
        self.image_source.rotation_radius = 0
        with ShouldRaise(ValueError("Sample off-centering too " +
                                    "small, enlarge rotation radius.")):
            self.aligner.run().wait()

    @slow
    def test_no_x_axis(self):
        """Test the case when there is no x-axis motor available."""
        self.align_check(17*q.deg, 11*q.deg, has_z_motor=False)

    @slow
    def test_not_misaligned(self):
        "Perfectly aligned rotation axis."
        self.align_check(0*q.deg, 0*q.deg)

    @slow
    def test_only_x(self):
        """Only misaligned laterally."""
        self.align_check(-17*q.deg, 0*q.deg)

    @slow
    def test_only_z(self):
        """Only misaligned in the beam direction."""
        self.align_check(0*q.deg, 11*q.deg)

    @slow
    def test_huge_x(self):
        self.image_source.scale = (3, 0.25, 3)
        self.align_check(60*q.deg, 11*q.deg)

    @slow
    def test_huge_z(self):
        self.image_source.scale = (3, 0.25, 3)
        self.align_check(11*q.deg, 60*q.deg)

    @slow
    def test_positive(self):
        self.align_check(17*q.deg, 11*q.deg)

    @slow
    def test_z_ambiguity(self):
        self.align_check(17*q.deg, -11*q.deg)

    @slow
    def test_negative_positive(self):
        self.align_check(-17*q.deg, 11*q.deg)

    @slow
    def test_positive_negative(self):
        self.align_check(17*q.deg, -11*q.deg)

    @slow
    def test_negative(self):
        self.align_check(-17*q.deg, -11*q.deg)
