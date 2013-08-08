import unittest
import logbook
from concert.quantities import q
from concert.tests import slow
from concert.measures.dummy import DummyGradientMeasure
from concert.devices.motors.dummy import Motor as DummyMotor, DummyLimiter
from concert.optimization.optimizers import Maximizer
from concert.optimization import algorithms


class TestDummyFocusing(unittest.TestCase):

    def setUp(self):
        self.handler = logbook.TestHandler()
        self.handler.push_application()
        self._motor = DummyMotor()
        self._motor.position = 0 * q.mm
        self._feedback = DummyGradientMeasure(
            self._motor['position'], 18.75 * q.mm)
        self.epsilon = 1 * q.um
        self.focuser = Maximizer(self._motor["position"], self._feedback,
                                 algorithms.halver,
                                (1 * q.mm,),
                                 {"epsilon": self.epsilon,
                                  "max_iterations": 1000})
        self._position_eps = 1e-1 * q.mm
        self._gradient_cmp_eps = 1e-1

    def tearDown(self):
        self.handler.pop_application()

    def _check_position(self, position, other_position):
        lower_bound_ok = other_position - self._position_eps <= position
        upper_bound_ok = position <= other_position + self._position_eps

        self.assertTrue(lower_bound_ok and upper_bound_ok,
                        "Motor position: %s " %
                        (str(position)) +
                        "differs more than by epsilon: %s " %
                        str(self._position_eps) +
                        "from the given position: %s" % (str(other_position)))

    @slow
    def test_maximum_in_limits(self):
        self.focuser.run().wait()
        self._check_position(self._motor.position,
                             self._feedback.max_position)

    @slow
    def test_huge_step_in_limits(self):
        self.focuser.step = 1000 * q.mm
        self.focuser.run().wait()
        self._check_position(self._motor.position,
                             self._feedback.max_position)

    @slow
    def test_maximum_out_of_limits_right(self):
        self._feedback.max_position = \
            (self._motor._hard_limits[1] + 50) * q.mm

        self.focuser.run().wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[1] * q.mm)

    @slow
    def test_maximum_out_of_limits_left(self):
        self._feedback.max_position = \
            (self._motor._hard_limits[0] - 50) * q.mm
        self.focuser.run().wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[0] * q.mm)

    @slow
    def test_maximum_out_of_soft_limits_right(self):
        motor = DummyMotor(limiter=DummyLimiter(25 * q.mm,
                                                75 * q.mm), position=50)
        feedback = DummyGradientMeasure(motor['position'], 80 * q.mm)
        focuser = Maximizer(motor["position"], feedback, algorithms.halver,
                            (motor.position,),
                            {"initial_step": 10 * q.mm,
                             "max_iterations": 1000})
        focuser.run().wait()

        self._check_position(motor.position, 75 * q.mm)

    @slow
    def test_maximum_out_of_soft_limits_left(self):
        motor = DummyMotor(limiter=DummyLimiter(25 * q.mm,
                                                75 * q.mm),
                           position=50)
        feedback = DummyGradientMeasure(motor['position'], 20 * q.mm)
        focuser = Maximizer(motor["position"], feedback, algorithms.halver,
                           (motor.position, 10 * q.mm, self.epsilon),
                            {"max_iterations": 1000})
        focuser._optimize()  # run().wait()
        self._check_position(motor.position, 25 * q.mm)

    @slow
    def test_huge_step_out_of_limits_right(self):
        # Right.
        self._feedback.max_position = (self._motor._hard_limits[1] + 50) * q.mm

        focuser = Maximizer(self._motor["position"], self._feedback,
                            algorithms.halver,
                           (self._motor.position, 1000 * q.mm, self.epsilon),
                            {"max_iterations": 1000})
        focuser.run().wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[1] * q.mm)

    @slow
    def test_huge_step_out_of_limits_left(self):
        focuser = Maximizer(self._motor["position"], self._feedback,
                            algorithms.halver,
                           (self._motor.position, 1000 * q.mm, self.epsilon),
                            {"max_iterations": 1000})
        self._feedback.max_position = (self._motor._hard_limits[0] - 50) * q.mm
        focuser.run().wait()
        self._check_position(self._motor.position,
                             self._motor._hard_limits[0] * q.mm)

    @slow
    def test_identical_gradients(self):
        # Some gradient level reached and then by moving to another position
        # the same level is reached again, but global gradient maximum
        # is not at this motor position.
        self._motor.position = -0.00001 * q.mm
        self.focuser.step = 10 * q.mm
        self.focuser.run().wait()
        self._check_position(self._motor.position,
                             self._feedback.max_position)
