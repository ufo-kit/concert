import numpy as np
from concert.async import async, wait
from concert.coroutines import coroutine
from concert.quantities import q
from concert.measures import get_rotation_axis
from concert.optimization import halver, optimize_parameter


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
        param.set(convert(xval)).join()
        yss.append(feedback())

    return (xss, yss)


def scan_param_feedback(scan_param, feedback_param,
                        minimum=None, maximum=None, intervals=64,
                        convert=lambda x: x):
    """Convenience function to scan one parameter and measure another.

    Scan the *scan_param* object and measure *feedback_param* at each of the
    *intervals* steps between *minimum* and *maximum*.

    Returns a tuple *(x, y)* with scanned parameter and measured values.
    """
    def feedback():
        return feedback_param.get().result()

    return scan(scan_param, feedback, minimum, maximum, intervals, convert)


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

    initialized_params = list(map(lambda x: x[0] + (x[1],),
                                  zip(param_list, initial_values)))

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


def focus(camera, motor, measure=np.std, opt_kwargs=None,
          plot_consumer=None, frame_consumer=None):
    """
    Focus *camera* by moving *motor*. *measure* is a callable that computes a
    scalar that has to be maximized from an image taken with *camera*.
    *opt_kwargs* are keyword arguments sent to the optimization algorithm.
    *plot_consumer* is fed with y values from the optimization and
    *frame_consumer* is fed with the incoming frames.

    This function is returning a future encapsulating the focusing event. Note,
    that the camera is stopped from recording as soon as the optimal position
    is found.
    """
    if opt_kwargs is None:
        opt_kwargs = {'initial_step': 0.5 * q.mm,
                      'epsilon': 1e-2 * q.mm}

    def get_measure():
        frame = camera.grab()
        if frame_consumer:
            frame_consumer.send(frame)
        return - measure(frame)

    @coroutine
    def filter_optimization():
        """
        Filter the optimization's (x, y) subresults to only the y part,
        otherwise the the plot update is not lucid.
        """
        while True:
            tup = yield
            if plot_consumer:
                plot_consumer.send(tup[1])

    camera.start_recording()
    f = optimize_parameter(motor['position'], get_measure, motor.position,
                           halver, alg_kwargs=opt_kwargs,
                           consumer=filter_optimization())
    f.add_done_callback(lambda unused: camera.stop_recording())
    return f


@async
def align_rotation_axis(camera, rotation_motor, flat_motor=None,
                        flat_position=None, x_motor=None, z_motor=None,
                        measure=get_rotation_axis, num_frames=10,
                        absolute_eps=0.1 * q.deg, max_iterations=5):
    """
    run(camera, rotation_motor, flat_motor, flat_position, x_motor=None,
    z_motor=None, measure=get_rotation_axis, num_frames=10, absolute_eps=0.1 *
    q.deg, max_iterations=5)

    Align rotation axis. *camera* is used to obtain frames, *rotation_motor*
    rotates the sample around the tomographic axis of rotation, *flat_motor* is
    used to move the sample out of the field of view in order to produce a flat
    field which will be used to correct the frame before segmentation.
    *flat_position* is the flat motor position in which the sample is out of
    the field of view. *x_motor* turns the sample around x-axis, *z_motor*
    turns the sample around z-axis.  *measure* provides axis of rotation
    angular misalignment data (a callable), *num_frames* defines how many
    frames are acquired and passed to the *measure*.  *absolute_eps* is the
    threshold for stopping the procedure. If *max_iterations* is reached the
    procedure stops as well.

    The procedure finishes when it finds the minimum angle between an
    ellipse extracted from the sample movement and respective axes or the
    found angle drops below *absolute_eps*. The axis of rotation after
    the procedure is (0,1,0), which is the direction perpendicular
    to the beam direction and the lateral direction.
    """
    if not x_motor and not z_motor:
        raise ValueError("At least one of the x, z motors must be given")

    step = 2 * np.pi / num_frames * q.rad

    flat = None
    if flat_motor:
        # First take a flat
        if flat_position is None:
            raise ValueError("If flat motor is given then also " +
                             "flat position must be given")
        flat_motor["position"].stash().join()
        flat_motor.position = flat_position
        flat = camera.grab().astype(np.float32)
        flat_motor["position"].restore().join()

    def get_frames():
        frames = []
        for i in range(num_frames):
            rotation_motor.move(i * step).join()
            frame = camera.grab()
            if flat:
                frame /= flat
            frames.append(frame)

        return frames

    # Sometimes both z-directions need to be tried out because of the
    # projection ambiguity.
    z_direction = -1
    i = 0

    x_last = None
    z_last = None
    z_turn_counter = 0

    while True:
        x_angle, z_angle, center = measure(get_frames())

        x_better = True if z_motor is not None and\
            (x_last is None or np.abs(x_angle) < x_last) else False
        z_better = True if x_motor is not None and\
            (z_last is None or np.abs(z_angle) < z_last) else False

        if x_motor:
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
            x_future = x_motor.move(z_direction * z_angle)
        if x_better and np.abs(x_angle) >= absolute_eps:
            z_future = z_motor.move(x_angle)
        elif (np.abs(z_angle) < absolute_eps or not z_better):
            # The newly calculated angles are worse than the previous
            # ones or the absolute threshold has been reached,
            # stop iteration.
            break

        wait([future for future in [x_future, z_future] if future is not None])

        x_last = np.abs(x_angle)
        z_last = np.abs(z_angle)

        i += 1

        if i == max_iterations:
            # If we reached maximum iterations we consider it a failure
            # because the algorithm was not able to get to the desired
            # solution within the max_iterations limit.
            raise ProcessException("Maximum iterations reached")

    # Return the last known ellipse fit
    return x_angle, z_angle, center


class ProcessException(Exception):

    """
    Exception raised by a process when something goes wrong with the procedure
    it tries to accomplish, e.g. cannot focus, cannot align rotation axis, etc.

    """

    pass
