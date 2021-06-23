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

    async def initialize(self):
        self.initialized = True

    async def measure(self):
        self.measured = True

    async def control(self):
        self.controlled = True

    async def compare(self):
        self.compared = True
        self.iteration += 1

        return self.iteration > 2


class TestOptimizationLoop(TestCase):
    def setUp(self):
        self.loop = VisitorLoop()

    async def test_all_called(self):
        await self.loop.run()
        self.assertTrue(self.loop.initialized)
        self.assertTrue(self.loop.measured)
        self.assertTrue(self.loop.controlled)

    async def test_run(self):
        self.assertFalse(await self.loop.run(max_iterations=1))
        self.assertTrue(await self.loop.run())
