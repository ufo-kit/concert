"""
The :mod:`.scan` module provides process classes and functions to change
parameters along pre-computed "trajectories".

:class:`Scanner` objects scan a parameter and evaluate a dependent variable ::

    from concert.processes.base import Scanner

    motor = Motor()
    camera = Camera()
    scanner = Scanner(motor['position'], lambda: camera.grab())
    scanner.minimum = 0 * q.mm
    scanner.maximum = 2 * q.mm

    x, y = scanner.run().result()

As you can see, the position of *motor* is varied and a frame at each interval
point is taken. Because processes run asynchronously, we call :meth:`result` on
the future that is returned by :meth:`~.Scanner.run`. This yields a tuple
with x and y values, corresponding to positions and frames.

For some tasks, feedbacks (such as the frame grabbing in the example above) are
pre-defined in the :mod:`concert.feedbacks.camera` module.

:func:`ascan` and :func:`dscan` are used to scan multiple parameters
in a similar way as SPEC::

    from concert.quantities import q
    from concert.processes.scan import ascan

    def do_something(parameters):
        for each parameter in parameters:
            print(parameter)

    ascan([(motor1['position'], 0 * q.mm, 25 * q.mm),
           (motor2['position'], -2 * q.cm, 4 * q.cm)],
           n_intervals=10, handler=do_something)

"""
import numpy as np
from concert.quantities import q
from concurrent.futures import wait
from concert.base import Parameter
from concert.asynchronous import async
from concert.processes.base import Process


class Scanner(Process):

    """A scan process.

    :meth:`.Scanner.run` sets *param* to :attr:`.intervals` data points and
    calls `feedback()` on each data point.

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
    def run(self, convert=lambda x: x):
        """run()

        Set :attr:`param` to values between :attr:`minimum` and
        :attr:`maximum`, call :attr:`feedback` on each data point and return a
        future with the value tuple.

        The result tuple *(x, y)* contains two array-likes with the same shape.
        *x* contains the values that :attr:`param` has taken during the scan
        whereas *y* contains the values evaluated by :attr:`feedback`.
        """
        xss = np.linspace(self.minimum, self.maximum, self.intervals)
        yss = []

        for xval in xss:
            self.param.set(convert(xval)).wait()
            yss.append(self.feedback())

        return (xss, yss)

    def show(self):
        """Call :meth:`run`, show the result of the scan with Matplotlib and
        return the plot object."""
        import matplotlib.pyplot as plt
        xval, yval = self.run().result()
        plt.xlabel(self.param.name)
        return plt.plot(xval, yval)


class StepTomoScanner(Process):

    """Tomo Scan Process."""

    def __init__(self, camera, rotary_stage,
                 prepare_dark_scan,
                 prepare_flat_scan,
                 prepare_proj_scan):

        params = [Parameter('angle', unit=q.deg)]

        self.camera = camera
        self.rotary_stage = rotary_stage
        self.prepare_dark_scan = prepare_dark_scan
        self.prepare_flat_scan = prepare_flat_scan
        self.prepare_proj_scan = prepare_proj_scan
        self.num_projections = 4

        super(StepTomoScanner, self).__init__(params)

    @async
    def run(self):
        def take_frames(prepare_step, n_frames=2):
            """Take frames."""
            frames = []
            prepare_step()
            self.camera.start_recording()

            for _ in xrange(n_frames):
                frames.append(self.camera.grab())

            self.camera.stop_recording()
            return frames

        darks = take_frames(self.prepare_dark_scan)
        flats = take_frames(self.prepare_flat_scan)

        projections = []
        step = self.angle

        self.prepare_proj_scan()
        self.camera.start_recording()

        for _ in xrange(self.num_projections):
            self.rotary_stage.move(step)
            projections.append(self.camera.grab())

        self.camera.stop_recording()
        return darks, flats, projections


def _pull_first(tuple_list):
    for tup in tuple_list:
        yield tup[0]


def ascan(param_list, n_intervals, handler, initial_values=None):
    """
    For each of the *n_intervals* and for each of the *(parameter, start,
    stop)* tuples in *param_list*, calculate a set value from *(stop - start) /
    n_intervals* and set *parameter* to it::

        ascan([(motor['position'], 0 * q.mm, 2 * q.mm)], 5, handler)

    When all devices have reached the set point *handler* is called with a list
    of the parameters as its first argument.

    If *initial_values* is given, it must be a list with the same length as
    *devices* containing start values from where each device is scanned.
    """
    parameters = [param for param in _pull_first(param_list)]

    if initial_values:
        if len(param_list) != len(initial_values):
            raise ValueError("*initial_values* must match *parameter_list*")
    else:
        initial_values = []

        for param in parameters:
            if param.unit:
                initial_values.append(0 * param.unit)
            else:
                initial_values.append(0)

    initialized_params = map(lambda (tup, single): tup + (single,),
                             zip(param_list, initial_values))

    for i in range(n_intervals + 1):
        futures = []

        for param, start, stop, init in initialized_params:
            step = (stop - start) / n_intervals
            value = init + start + i * step
            futures.append(param.set(value))

        wait(futures)
        handler(parameters)


def dscan(parameter_list, n_intervals, handler):
    """
    For each of the *n_intervals* and for each of the *(parameter, start,
    stop)* tuples in *param_list*, calculate a set value from *(stop - start) /
    n_intervals* and set *parameter*.
    """
    initial_values = [param.get().result()
                      for param in _pull_first(parameter_list)]

    return ascan(parameter_list, n_intervals, handler, initial_values)
