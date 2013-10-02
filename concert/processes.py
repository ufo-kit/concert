import numpy as np
from concert.base import Process
from concert.helpers import async, wait
from concert.quantities import q
from concert.optimization.algorithms import halver
from concert.optimization.optimizers import Maximizer


def _pull_first(tuple_list):
        for tup in tuple_list:
            yield tup[0]




@async
def scan(param, feedback, minimum=None, maximum=None, intervals=64,
         convert=lambda x: x):
    """Scan a parameter and provide a feedback.

    Scan the parameter object in *intervals* steps between *minimum* and
    *maximum* and call *feedback* at each step. *feedback* must return a value
    that is evaluated at the parameter position. If *minimum* or *maximum* is
    ``None``, :attr:`Parameter.lower` or :attr:`Parameter.upper` is used.

    Set *convert* to a callable that transforms the parameter value prior to
    setting it.

    Returns a tuple *(x, y)* with parameter and feedback values.
    """
    minimum = minimum if minimum is not None else param.lower
    maximum = maximum if maximum is not None else param.upper

    xss = np.linspace(minimum, maximum, intervals)
    yss = []

    for xval in xss:
        param.set(convert(xval)).wait()
        yss.append(feedback())

    return (xss, yss)


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


def focus(camera, motor, measure=np.std):
    """
    Focus *camera* by moving *motor*. *measure* is a callable that computes a
    scalar that has to be maximized from an image taken with *camera*.

    This function is returning a future encapsulating the focusing event. Note,
    that the camera is stopped from recording as soon as the optimal position
    is found.
    """
    # we should guess this from motor limits
    opts = {'initial_step': 10 * q.mm,
            'epsilon': 5e-3 * q.mm}

    def get_measure():
        frame = camera.grab()
        return measure(frame)

    maximizer = Maximizer(motor['position'],
                          get_measure,
                          halver, alg_kwargs=opts)

    camera.start_recording()
    f = maximizer.run()
    f.add_done_callback(lambda unused: camera.stop_recording())
    return f


class RotationAxisAligner(Process):

    """Rotation axis alignment."""
    # Aligned message
    AXIS_ALIGNED = "axis-aligned"

    def __init__(self, axis_measure, get_images, x_motor, z_motor=None):
        """Contructor. *axis_measure* provides axis of rotation angular
        misalignment data, *get_images* provides image sequences with
        sample rotated around axis of rotation (it is a callable).
        *x_motor* turns the sample around x-axis, *z_motor* is optional
        and turns the sample around z-axis
        """
        super(RotationAxisAligner, self).__init__(None)
        self._axis_measure = axis_measure
        self.x_motor = x_motor
        self.z_motor = z_motor
        self.get_images = get_images

    @async
    def run(self, absolute_eps=0.1 * q.deg):
        """
        run(absolute_eps=0.1*q.deg)

        The procedure finishes when it finds the minimum angle between an
        ellipse extracted from the sample movement and respective axes or the
        found angle drops below *absolute_eps*. The axis of rotation after
        the procedure is (0,1,0), which is the direction perpendicular
        to the beam direction and the lateral direction.
        """
        # Sometimes both z-directions need to be tried out because of the
        # projection ambiguity.
        z_direction = -1

        x_last = None
        z_last = None
        z_turn_counter = 0

        while True:
            self._axis_measure.images = self.get_images()
            x_angle, z_angle = self._axis_measure()

            x_better = True if self.z_motor is not None and\
                (x_last is None or np.abs(x_angle) < x_last) else False
            z_better = True if z_last is None or np.abs(z_angle) < z_last\
                else False

            if z_better:
                z_turn_counter = 0
            elif z_turn_counter < 1:
                # We might have rotated in the opposite direction because
                # of the projection ambiguity. However, that must be picked up
                # in the next iteration, so if the two consequent angles
                # are worse than the minimum, we have the result.
                z_better = True
                z_direction = -z_direction
                z_turn_counter += 1

            x_future, z_future = None, None
            if z_better and np.abs(z_angle) >= absolute_eps:
                x_future = self.x_motor.move(z_direction * z_angle)
            if x_better and np.abs(x_angle) >= absolute_eps:
                z_future = self.z_motor.move(x_angle)
            elif (np.abs(z_angle) < absolute_eps or not z_better):
                # The newly calculated angles are worse than the previous
                # ones or the absolute threshold has been reached,
                # stop iteration.
                dispatcher.send(self, self.AXIS_ALIGNED)
                break

            wait([future for future in [x_future, z_future]
                  if future is not None])

            x_last = np.abs(x_angle)
            z_last = np.abs(z_angle)
