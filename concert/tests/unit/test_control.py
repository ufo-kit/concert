from concert.experiments.control import ClosedLoop
from concert.tests import TestCase


class VisitorLoop(ClosedLoop):

    """A visitor loop checks visitations of control loop nodes. On the first run, it simulates
    unsuccessful result, the second run is already a success.
    """

    def __init__(self):
        super(VisitorLoop, self).__init__()
        self.initialized = False
        self.measured = False
        self.compared = False
        self.controlled = False
        self.iteration = 1

    def initialize(self):
        self.initialized = True

    def measure(self):
        self.measured = True

    def control(self):
        self.controlled = True

    def compare(self):
        self.compared = True
        self.iteration += 1

        return self.iteration > 2


class TestOptimizationLoop(TestCase):
    def setUp(self):
        self.loop = VisitorLoop()

    def test_all_called(self):
        self.loop.run().join()
        self.assertTrue(self.loop.initialized)
        self.assertTrue(self.loop.measured)
        self.assertTrue(self.loop.controlled)

    def test_run(self):
        self.assertFalse(self.loop.run(max_iterations=1).result())
        self.assertTrue(self.loop.run().result())
