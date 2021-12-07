"""Experiment automation based on on-line data analysis."""
from concert.coroutines.base import background


class ClosedLoop(object):
    """An abstract feedback loop which acquires data, analyzes it on-line and provides feedback to
    the experiment. The data acquisition procedure is done iteratively until the result of some
    metric converges to a satisfactory value. Schematically, the class is doing the following in an
    iterative way::

        initialize -> measure -> compare -> OK -> success
                        ^            |
                        |           NOK
                        |            |
                        -- control <--

    """
    async def initialize(self):
        """Bring the experimental setup to some defined initial (reference) state."""
        pass

    async def measure(self):
        """Conduct a measurement from data acquisition to analysis."""
        pass

    async def control(self):
        """React on the result of a measurement."""
        pass

    async def compare(self):
        """Return True if the metric is satisfied, False otherwise. This is the decision making
        process.
        """
        raise NotImplementedError

    @background
    async def run(self, max_iterations=10):
        """
        run(self, max_iterations=10)

        Run the loop until the metric is satisfied, if we don't converge in *max_iterations* then
        the run is considered unsuccessful and False is returned, otherwise True.
        """
        await self.initialize()

        for i in range(max_iterations):
            await self.measure()
            if await self.compare():
                return True
            else:
                await self.control()

        return False


class DummyLoop(ClosedLoop):

    """A dummy optimization loop."""

    async def compare(self):
        return True
