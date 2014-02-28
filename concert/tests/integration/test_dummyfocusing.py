from concert import optimization
from concert.base import HardLimitError
from concert.quantities import q
from concert.tests import slow, assert_almost_equal, TestCase
from concert.measures import DummyGradientMeasure
from concert.devices.motors.dummy import LinearMotor
from concert.optimization import optimize_parameter


class TestDummyFocusingWithSoftLimits(TestCase):

    def setUp(self):
        super(TestDummyFocusingWithSoftLimits, self).setUp()
        self.motor = LinearMotor(position=50 * q.mm)
        self.motor['position'].lower = 25 * q.mm
        self.motor['position'].upper = 75 * q.mm
        self.halver_kwargs = {"initial_step": 10 * q.mm,
                              "max_iterations": 1000}

    @slow
    def test_maximum_out_of_soft_limits_right(self):
        feedback = DummyGradientMeasure(self.motor['position'], 80 * q.mm)
        optimize_parameter(self.motor["position"], lambda: - feedback(),
                           self.motor.position, optimization.halver,
                           alg_kwargs=self.halver_kwargs).join()
        assert_almost_equal(self.motor.position, 75 * q.mm)

    @slow
    def test_maximum_out_of_soft_limits_left(self):
        feedback = DummyGradientMeasure(self.motor['position'], 20 * q.mm)
        optimize_parameter(self.motor["position"], lambda: - feedback(),
                           self.motor.position, optimization.halver,
                           alg_kwargs=self.halver_kwargs).join()
        assert_almost_equal(self.motor.position, 25 * q.mm)


class TestDummyFocusing(TestCase):

    def setUp(self):
        super(TestDummyFocusing, self).setUp()
        self.motor = LinearMotor()
        self.motor.position = 0 * q.mm
        self.feedback = DummyGradientMeasure(self.motor['position'],
                                             18.75 * q.mm)
        self.epsilon = 1 * q.um
        self.position_eps = 1e-1 * q.mm
        self.gradient_cmp_eps = 1e-1
        self.halver_kwargs = {"initial_step": 10 * q.mm,
                              "max_iterations": 3000}

    def run_optimization(self, initial_position=1 * q.mm):
        optimize_parameter(self.motor["position"], lambda: - self.feedback(),
                           initial_position, optimization.halver,
                           alg_kwargs=self.halver_kwargs).join()

    def check(self, other):
        assert_almost_equal(self.motor.position, other, 0.1)

    @slow
    def test_maximum_in_limits(self):
        self.run_optimization()
        self.check(self.feedback.max_position)

    @slow
    def test_huge_step_in_limits(self):
        self.halver_kwargs["initial_step"] = 1000 * q.mm

        with self.assertRaises(HardLimitError):
            self.run_optimization()

    @slow
    def test_maximum_out_of_limits_right(self):
        self.feedback.max_position = (self.motor.upper + 50 * q.mm)

        with self.assertRaises(HardLimitError):
            self.run_optimization()

    @slow
    def test_maximum_out_of_limits_left(self):
        self.feedback.max_position = (self.motor.lower - 50 * q.mm)

        with self.assertRaises(HardLimitError):
            self.run_optimization()

    @slow
    def test_identical_gradients(self):
        # Some gradient level reached and then by moving to another position
        # the same level is reached again, but global gradient maximum
        # is not at this motor position.
        self.halver_kwargs["initial_step"] = 10 * q.mm
        self.run_optimization(initial_position=-0.00001 * q.mm)
        self.check(self.feedback.max_position)
