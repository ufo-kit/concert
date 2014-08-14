"""Experiment automation based on on-line data analysis."""
from concert.async import async


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
    def initialize(self):
        """Bring the experimental setup to some defined initial (reference) state."""
        pass

    def measure(self):
        """Conduct a measurement from data acquisition to analysis."""
        pass

    def control(self):
        """React on the result of a measurement."""
        pass

    def compare(self):
        """Return True if the metric is satisfied, False otherwise. This is the decision making
        process.
        """
        raise NotImplementedError

    @async
    def run(self, max_iterations=10):
        """
        run(self, max_iterations=10)

        Run the loop until the metric is satisfied, if we don't converge in *max_iterations* then
        the run is considered unsuccessful and False is returned, otherwise True.
        """
        self.initialize()

        for i in range(max_iterations):
            self.measure()
            if self.compare():
                return True
            else:
                self.control()

        return False


class DummyLoop(ClosedLoop):

    """A dummy optimization loop."""

    def compare(self):
        return True
