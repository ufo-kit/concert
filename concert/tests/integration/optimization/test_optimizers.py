import unittest
import logbook
from concert.quantities import q
from concert.tests import assert_almost_equal
from concert.devices.motors.dummy import Motor
from concert.optimization import algorithms as algs
from concert.optimization.optimizers import Minimizer, Maximizer
from concert.tests import slow


class TestOptimizers(unittest.TestCase):

    def setUp(self):
        self.handler = logbook.TestHandler()
        self.handler.push_application()
        self.algorithms = [algs.halver, algs.down_hill, algs.powell,
                           algs.nonlinear_conjugate, algs.bfgs,
                           algs.least_squares]
        self.center = 3.0 * q.mm
        self.motor = Motor(position=0)

    def tearDown(self):
        self.handler.pop_application()

    def feedback(self):
        return (self.motor.position.to_base_units().magnitude -
                self.center.to_base_units().magnitude) ** 2

    def check(self):
        assert_almost_equal(self.motor.position.to_base_units(),
                            self.center.to_base_units(), 1e-2)

    @slow
    def test_minimizers(self):
        for i in range(len(self.algorithms)):
            self.motor.position = 0 * q.mm
            minim = Minimizer(self.motor["position"], self.feedback,
                              self.algorithms[i])
            minim.run().wait()
            self.check()

    @slow
    def test_maximizers(self):
        for i in range(len(self.algorithms)):
            self.motor.position = 0 * q.mm
            minim = Maximizer(self.motor["position"],
                              lambda: - self.feedback(), self.algorithms[i])
            minim.run().wait()
            self.check()
