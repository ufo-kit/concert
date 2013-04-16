"""This module provides processing capabilities.

Just like a :class:`.Device`, a :class:`Process` is a :class:`.Parameterizable`
object with a generic method :meth:`~Process.run` that is executed
asynchronously.
"""

import numpy as np
from concert.base import Parameterizable, Parameter
from concert.asynchronous import async


class Process(Parameterizable):
    """Base process."""
    def __init__(self, params):
        super(Process, self).__init__(params)

    @async
    def run(self):
        """run()

        Run the process. The result depends on the actual process.
        """
        raise NotImplementedError


class Scanner(Process):
    """A scan process.

    :meth:`.Scanner.run` sets *param* to :attr:`.intervals` data points and
    calls `feedback()` on each data point::

        motor = Motor()
        camera = Camera()
        scanner = Scanner(motor['position'], lambda: camera.grab())
        x, y = scanner.run().result()

    .. py:attribute:: param

        The scanned :class:`.Parameter`. It must have the same unit as
        :attr:`minimum` and :attr:`maximum`.

    .. py:attribute:: feedback

        A callable that must return a scalar value. It is called for each data
        point.

    .. py:attribute:: minimum

        The lower bound of the scannable range.

    .. py:attribute:: maximum

        The upper bound of the scannable range.

    .. py:attribute:: intervals

        The number of intervals that are scanned.
    """

    def __init__(self, param, feedback):
        params = [Parameter('minimum', doc="Left bound of the interval"),
                  Parameter('maximum', doc="Right bound of the interval"),
                  Parameter('intervals', doc="Number of intervals")]

        super(Scanner, self).__init__(params)
        self.intervals = 64
        self.param = param
        self.feedback = feedback

    @async
    def run(self):
        """run()

        Set :attr:`param` to values between :attr:`minimum` and :attr:`maximum`
        and call :attr:`feedback` on each data point.
        """

        xs = np.linspace(self.minimum, self.maximum, self.intervals)
        ys = np.zeros(xs.shape)

        for i, x in enumerate(xs):
            self._param.set(x).wait()
            ys[i] = self._feedback()

        return (xs, ys)

    def show(self):
        """Call :meth:`run` and show the result of the scan with Matplotlib.

        The method returns the plot object.
        """
        import matplotlib.pyplot as plt
        x, y = self.run().result()
        plt.xlabel(self._param.name)
        return plt.plot(x, y)
