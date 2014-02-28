from concert.quantities import q
from concert import optimization
from concert.tests import slow, assert_almost_equal, TestCase
from concert.optimization import optimize_parameter
from concert.devices.motors.dummy import LinearMotor


class TestOptimizers(TestCase):

    def setUp(self):
        super(TestOptimizers, self).setUp()
        self.algorithms = [optimization.halver, optimization.down_hill,
                           optimization.powell,
                           optimization.nonlinear_conjugate,
                           optimization.bfgs,
                           optimization.least_squares]
        self.center = 1.0 * q.mm
        self.motor = LinearMotor(position=0 * q.mm)
        self.motor.lower = -float('Inf') * q.mm
        self.motor.upper = float('Inf') * q.mm

    def feedback(self):
        return (self.motor.position.to(q.mm).magnitude -
                self.center.to(q.mm).magnitude) ** 2

    def check(self):
        assert_almost_equal(self.motor.position.to(q.mm),
                            self.center.to(q.mm), 1e-2)

    def optimize(self, algorithm):
        optimize_parameter(self.motor["position"], self.feedback,
                           self.motor.position, algorithm).join()

    @slow
    def test_algorithms(self):
        for i in range(len(self.algorithms)):
            self.motor.position = 0 * q.mm
            self.optimize(self.algorithms[i])
            self.check()
