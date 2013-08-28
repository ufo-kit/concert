import unittest
import logbook
from concert.quantities import q
from concert.tests import slow, assert_almost_equal
from concert.measures.dummy import DummyGradientMeasure
from concert.devices.motors.dummy import Motor as DummyMotor
from concert.optimization.optimizers import Maximizer
from concert.optimization import algorithms


class TestDummyFocusingWithSoftLimits(unittest.TestCase):

    def setUp(self):
        self.handler = logbook.NullHandler()
        self.handler.push_application()
        self.motor = DummyMotor(position=50 * q.count)
        self.motor['position'].lower = 25 * q.mm
        self.motor['position'].upper = 75 * q.mm
        self.focuser = Maximizer(self.motor["position"], None,
                                 algorithms.halver,
                                 (self.motor.position,),
                                 {"initial_step": 10 * q.mm,
                                  "max_iterations": 1000})

    @slow
    def test_maximum_out_of_soft_limits_right(self):
        feedback = DummyGradientMeasure(self.motor['position'], 80 * q.mm)
        self.focuser.feedback = feedback
        self.focuser.run().wait()
        assert_almost_equal(self.motor.position, 25 * q.mm)

    @slow
    def test_maximum_out_of_soft_limits_left(self):
        feedback = DummyGradientMeasure(self.motor['position'], 20 * q.mm)
        self.focuser.feedback = feedback
        self.focuser.run().wait()
        assert_almost_equal(self.motor.position, 75 * q.mm)


class TestDummyFocusing(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.handler = logbook.NullHandler()
        self.handler.push_application()
        self.motor = DummyMotor()
        self.motor.position = 0 * q.mm
        self.feedback = DummyGradientMeasure(self.motor['position'],
                                             18.75 * q.mm)
        self.epsilon = 1 * q.um
        self.focuser = Maximizer(self.motor["position"], self.feedback,
                                 algorithms.halver,
                                 (1 * q.mm,),
                                 {"epsilon": self.epsilon,
                                  "max_iterations": 1000})
        self.position_eps = 1e-1 * q.mm
        self.gradient_cmp_eps = 1e-1

    def tearDown(self):
        self.handler.pop_application()

    def _check_position(self, position, other_position):
        lower_bound_ok = other_position - self.position_eps <= position
        upper_bound_ok = position <= other_position + self.position_eps

        self.assertTrue(lower_bound_ok and upper_bound_ok,
                        "Motor position: %s " %
                        (str(position)) +
                        "differs more than by epsilon: %s " %
                        str(self.position_eps) +
                        "from the given position: %s" % (str(other_position)))

    def check(self, other):
        assert_almost_equal(self.motor.position, other, 0.1)

    @slow
    def test_maximum_in_limits(self):
        self.focuser.run().wait()
        self.check(self.feedback.max_position)

    @slow
    def test_huge_step_in_limits(self):
        self.focuser.step = 1000 * q.mm
        self.focuser.run().wait()
        self.check(self.feedback.max_position)

    @slow
    def test_maximum_out_of_limits_right(self):
        self.feedback.max_position = (self.motor.upper + 50) * q.mm

        self.focuser.run().wait()
        self.check(self.motor.upper * q.mm)

    @slow
    def test_maximum_out_of_limits_left(self):
        self.feedback.max_position = (self.motor.lower - 50) * q.mm
        self.focuser.run().wait()
        self.check(self.motor.lower * q.mm)

    @slow
    def test_huge_step_out_of_limits_right(self):
        # Right.
        self.feedback.max_position = (self.motor.upper + 50) * q.mm

        focuser = Maximizer(self.motor["position"], self.feedback,
                            algorithms.halver,
                           (self.motor.position, 1000 * q.mm, self.epsilon),
                            {"max_iterations": 1000})
        focuser.run().wait()
        self.check(self.motor.upper * q.mm)

    @slow
    def test_huge_step_out_of_limits_left(self):
        focuser = Maximizer(self.motor["position"], self.feedback,
                            algorithms.halver,
                           (self.motor.position, 1000 * q.mm, self.epsilon),
                            {"max_iterations": 1000})
        self.feedback.max_position = (self.motor.lower - 50) * q.mm
        focuser.run().wait()
        self.check(self.motor.lower * q.mm)

    @slow
    def test_identical_gradients(self):
        # Some gradient level reached and then by moving to another position
        # the same level is reached again, but global gradient maximum
        # is not at this motor position.
        self.motor.position = -0.00001 * q.mm
        self.focuser.step = 10 * q.mm
        self.focuser.run().wait()
        self.check(self.feedback.max_position)
