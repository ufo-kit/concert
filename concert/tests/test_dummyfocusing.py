import unittest
import logbook
import quantities as q
import random
from concert.tests import slow
from concert.devices.motors.base import LinearCalibration
from concert.measures.dummygradient import DummyGradientMeasure
from concert.processes.focus import Focuser
from concert.devices.motors.dummy import DummyMotor, DummyLimiter


class TestDummyFocusing(unittest.TestCase):
    def setUp(self):
        self._motor = DummyMotor(LinearCalibration(1/q.mm, 0*q.mm))
        self._gradient_feedback = DummyGradientMeasure(self._motor, 18.75*q.mm)
        self._focuser = Focuser(self._motor, 1e-3,
                                self._gradient_feedback.get_gradient)
        self._position_eps = 1e-1*q.mm
        self._gradient_cmp_eps = 1e-1
        self.handler = logbook.TestHandler()
        self.handler.push_application()

    def tearDown(self):
        self.handler.pop_application()

    def _check_position(self, position, other_position):
        self.assertTrue(other_position -
                        self._position_eps <= position <=
                        other_position + self._position_eps,
                        "Motor position: %s " %
                        (str(position)) +
                        "differs more than by epsilon: %g " %
                        (self._position_eps) +
                        "from the given position: %s" % (str(other_position)))

    @slow
    def test_maximum_in_limits(self):
        self._focuser.focus(1*q.mm).wait()
        self._check_position(self._motor.position,
                             self._gradient_feedback.max_gradient_position)

    @slow
    def test_huge_step_in_limits(self):
        self._focuser.focus(1000*q.mm).wait()
        self._check_position(self._motor.position,
                             self._gradient_feedback.max_gradient_position)

    @slow
    def test_maximum_out_of_limits(self):
        # Right.
        self._gradient_feedback.max_gradient_position = \
            (self._motor._hard_limits[1]+50)*q.mm
        self._focuser.focus(1*q.mm).wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[1]*q.mm)

        # Left.
        self._gradient_feedback.max_gradient_position = \
            (self._motor._hard_limits[0]-50)*q.mm
        self._focuser.focus(1*q.mm).wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[0]*q.mm)

    @slow
    def test_maximum_out_of_soft_limits(self):
        # Right.
        motor = DummyMotor(LinearCalibration(1/q.mm, 0*q.mm),
                           limiter=DummyLimiter(25, 75),
                           position=random.uniform(25, 75))
        gradient_feedback = DummyGradientMeasure(motor, 80*q.mm)
        focuser = Focuser(motor, 1e-3, gradient_feedback.get_gradient)
        focuser.focus(10*q.mm).wait()
        self._check_position(motor.position, 75*q.mm)

        # Left.
        gradient_feedback = DummyGradientMeasure(motor, 20*q.mm)
        focuser = Focuser(motor, 1e-3, gradient_feedback.get_gradient)
        focuser.focus(10*q.mm).wait()
        self._check_position(motor.position, 25*q.mm)

    @slow
    def test_huge_step_out_of_limits(self):
        # Right.
        self._gradient_feedback.max_gradient_position = \
            (self._motor._hard_limits[1]+50)*q.mm
        focuser = Focuser(self._motor, 1e-3,
                          self._gradient_feedback.get_gradient)
        focuser.focus(1000*q.mm).wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[1]*q.mm)

        # Left.
        self._gradient_feedback.max_gradient_position = \
            (self._motor._hard_limits[0]-50)*q.mm
        focuser = Focuser(self._motor, 1e-3,
                          self._gradient_feedback.get_gradient)
        focuser.focus(1000*q.mm).wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[0]*q.mm)

    @slow
    def test_identical_gradients(self):
        # Some gradient level reached and then by moving to another position
        # the same level is reached again, but global gradient maximum
        # is not at this motor position.
        self._motor.position = -0.00001*q.mm
        self._focuser.focus(10*q.mm).wait()
        self._check_position(self._motor.position,
                             self._gradient_feedback.max_gradient_position)
