from concert.devices.motors.dummy import Motor
from concert.optimization import algorithms as algs
from concert.optimization.optimizers import Minimizer, Maximizer
import logbook
import quantities as q
import unittest
from concert.tests import slow


class TestOptimizers(unittest.TestCase):

    def setUp(self):
        self.handler = logbook.TestHandler()
        self.handler.push_application()
        self.algorithms = [algs.halver, algs.down_hill, algs.powell,
                           algs.nonlinear_conjugate, algs.bfgs,
                           algs.least_squares]
        self.center = 3.0 * q.mm
        self.precision_places = 2
        self.motor = Motor(position=0)

    def tearDown(self):
        self.handler.pop_application()

    def feedback(self):
        return (self.motor.position.simplified.magnitude -
                self.center.simplified.magnitude) ** 2

    def check(self):
        self.assertAlmostEqual(self.motor.position.simplified,
                               self.center.simplified, self.precision_places)

    @slow
    def test_minimizers(self):
        for i in range(len(self.algorithms)):
            self.motor.position = 0 * q.mm
            args = (self.motor["position"], ) if i == 0 else\
                (self.motor.position, )

            minim = Minimizer(self.motor["position"], self.feedback,
                              self.algorithms[i], args)
            minim.run().wait()
            self.check()

    @slow
    def test_maximizers(self):
        for i in range(len(self.algorithms)):
            self.motor.position = 0 * q.mm
            args = (self.motor["position"], ) if i == 0 else\
                (self.motor.position, )

            minim = Maximizer(self.motor["position"],
                              lambda: - self.feedback(),
                              self.algorithms[i], args)
            minim.run().wait()
            self.check()
