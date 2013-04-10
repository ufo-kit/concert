import numpy as np
from concert.base import Parameterizable, Parameter
from concert.asynchronous import async


class Process(Parameterizable):
    """Base process."""
    def __init__(self, params):
        super(Process, self).__init__(params)

    @async
    def run(self):
        """Run the process.

        The result depends on the actual process.
        """
        raise NotImplementedError


class Feedback(object):
    """Provide a feedback."""

    def evaluate(self):
        """This must return a value. Any kind of value."""
        raise NotImplementedError


class Scanner(Process):
    """A scanner process.

    Calling :meth:`.run` will set *param* to *intervals* data points and
    call :meth:`.evaluate`.
    """

    def __init__(self, param, feedback):
        params = [Parameter('minimum', doc="Left bound of the interval"),
                  Parameter('maximum', doc="Right bound of the interval"),
                  Parameter('intervals', doc="Number of intervals")]

        super(Scanner, self).__init__(params)
        self._param = param
        self._feedback = feedback

    @async
    def run(self):
        xs = np.linspace(self.minimum, self.maximum, self.intervals)
        ys = np.zeros(xs.shape)

        for i, x in enumerate(xs):
            self._param.set(x).wait()
            ys[i] = self._feedback.evaluate()

        return (xs, ys)

    def show(self):
        """Show the result of the scan with Matplotlib."""
        import matplotlib.pyplot as plt
        x, y = self.run().result()
        plt.xlabel(self._param.name)
        plt.plot(x, y)

    def evaluate(self):
        raise NotImplementedError
