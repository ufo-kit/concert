from itertools import product
import numpy as np
import logging
from scipy.ndimage.filters import gaussian_filter
from concert.casync import casync, wait
from concert.quantities import q
from concert.measures import rotation_axis
from concert.optimization import halver, optimize_parameter
from concert.imageprocessing import center_of_mass, flat_correct, find_needle_tips
from concert.coroutines.base import coroutine
from concert.helpers import expects, Numeric
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.devices.shutters.base import Shutter
from concert.devices.cameras.base import Camera
from concert.progressbar import wrap_iterable


LOG = logging.getLogger(__name__)


def _pull_first(tuple_list):
        for tup in tuple_list:
            yield tup[0]


def scan(feedback, regions, callbacks=None):
    """A multidimensional scan. *feedback* is a callable which takes no arguments and provides
    feedback after some parameter is changed. *regions* specifies the scanned parameter, it is
    either a :class:`concert.helpers.Region` or a list of those for multidimensional scan. The
    fastest changing parameter is the last one specified. *callbacks* is a dictionary in the form
    {region: function}, where *function* is a callable with no arguments (just like *feedback*) and
    is called every time the parameter in *region* is changed. One would use a scan for example like
    this::

        import numpy as np
        from concert.casync import resolve
        from concert.helpers import Region

        def take_flat_field():
            # Do something here
            pass

        exp_region = Region(camera['exposure_time'], np.linspace(1, 100, 100) * q.ms)
        position_region = Region(motor['position'], np.linspace(0, 180, 1000) * q.deg)
        callbacks = {exp_region: take_flat_field}

        # This is a 2D scan with position_region in the inner loop. It acquires a tomogram, changes
        # the exposure time and continues like this until all exposure times are exhausted.
        # Take_flat_field is called every time the exposure_time of the camera is changed
        # (in this case after every tomogram) and you can use it to correct the acquired images.
        for result in resolve(scan(camera.grab, [exp_region, position_region],
                              callbacks=callbacks)):
            # Do something real instead of just a print
            print result

    From the execution order it is equivalent to (in reality there is more for making the code
    casynchronous)::

        for exp_time in np.linspace(1, 100, 100) * q.ms:
            for position in np.linspace(0, 180, 1000) * q.deg:
                yield feedback()

    """
    changes = []
    if not isinstance(regions, (list, tuple, np.ndarray)):
        regions = [regions]

    if callbacks is None:
        callbacks = {}

    # Changes store the indices at which parameters change, e.g. for two parameters and interval
    # lengths 2 for first and 3 for second changes = [3, 1], i. e. first parameter is changed when
    # the flattened iteration index % 3 == 0, second is changed every iteration.
    # we do this because parameter setting might be expensive even if the value does not change
    current_mul = 1
    for i in range(len(regions))[::-1]:
        changes.append(current_mul)
        current_mul *= len(regions[i].values)
    changes.reverse()

    def get_changed(index):
        """Returns a tuple of indices of changed parameters at given iteration *index*."""
        return [i for i in range(len(regions)) if index % changes[i] == 0]

    @casync
    def get_value(index, tup, previous):
        """Get value after setting parameters, *index* is the flattened iteration index, *tup* are
        all the parameter values, *previous* is the previous future or None if this is the first
        time.
        """
        if previous:
            previous.join()

        changed = get_changed(index)
        futures = []
        for i in changed:
            futures.append(regions[i].parameter.set(tup[i]))
        wait(futures)

        for i in changed:
            if regions[i] in callbacks:
                callbacks[regions[i]]()

        return tup + (feedback(),)

    future = None

    for i, tup in wrap_iterable(enumerate(product(*regions))):
        future = get_value(i, tup, future)
        yield future


def scan_param_feedback(scan_param_regions, feedback_param, callbacks=None):
    """
    Convenience function to scan some parameters and measure another parameter.

    Scan the *scan_param_regions* parameters and measure *feedback_param*.
    """
    def feedback():
        return feedback_param.get().result()

    return scan(feedback, scan_param_regions, callbacks=callbacks)


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

    initialized_params = list([x[0] + (x[1],) for x in zip(param_list, initial_values)])

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


@expects(Camera, LinearMotor, measure=None, opt_kwargs=None,
         plot_consumer=None, frame_consumer=None)
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
        camera.trigger()
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

    camera.trigger_source = camera.trigger_sources.SOFTWARE
    camera.start_recording()
    f = optimize_parameter(motor['position'], get_measure, motor.position,
                           halver, alg_kwargs=opt_kwargs,
                           consumer=filter_optimization())
    f.add_done_callback(lambda unused: camera.stop_recording())
    return f


@casync
@expects(Camera, RotationMotor, num_frames=Numeric(1), shutter=Shutter,
         flat_motor=LinearMotor, flat_position=Numeric(1, q.m), y_0=Numeric(1),
         y_1=Numeric(1), frame_consumer=None)
def acquire_frames_360(camera, rotation_motor, num_frames, shutter=None, flat_motor=None,
                       flat_position=None, y_0=0, y_1=None, frame_consumer=None):
    """
    acquire_frames_360(camera, rotation_motor, num_frames, shutter=None, flat_motor=None,
                       flat_position=None, y_0=0, y_1=None, frame_consumer=None)

    Acquire frames around a circle.
    """
    frames = []
    flat = None
    if camera.state == 'recording':
        camera.stop_recording()
    camera['trigger_source'].stash().join()
    camera.trigger_source = camera.trigger_sources.SOFTWARE

    try:
        camera.start_recording()
        if shutter:
            if shutter.state != 'closed':
                shutter.close().join()
            camera.trigger()
            dark = camera.grab()[y_0:y_1]
            shutter.open().join()
        if flat_motor and flat_position is not None:
            radio_position = flat_motor.position
            flat_motor.position = flat_position
            camera.trigger()
            flat = camera.grab()[y_0:y_1]
            flat_motor.position = radio_position
        for i in range(num_frames):
            rotation_motor.move(2 * np.pi / num_frames * q.rad).join()
            camera.trigger()
            frame = camera.grab()[y_0:y_1].astype(np.float)
            if flat is not None:
                frame = flat_correct(frame, flat, dark=dark)
                frame = np.nan_to_num(-np.log(frame))
                # Huge numbers can also cause trouble
                frame[np.abs(frame) > 1e6] = 0
            else:
                frame = frame.max() - frame
            if frame_consumer:
                frame_consumer.send(frame)
            frames.append(frame)

        return frames
    except Exception as e:
        LOG.error(e)
        raise
    finally:
        camera.stop_recording()
        # No side effects
        camera['trigger_source'].restore().join()


@casync
@expects(Camera, RotationMotor, x_motor=RotationMotor, z_motor=RotationMotor,
         get_ellipse_points=find_needle_tips, num_frames=Numeric(1), metric_eps=Numeric(1, q.deg),
         position_eps=Numeric(1, q.deg), max_iterations=Numeric(1),
         initial_x_coeff=Numeric(1, q.dimensionless), initial_z_coeff=Numeric(1, q.dimensionless),
         shutter=Shutter, flat_motor=LinearMotor, flat_position=Numeric(1, q.m),
         y_0=Numeric(1), y_1=Numeric(1), get_ellipse_points_kwargs=None, frame_consumer=None)
def align_rotation_axis(camera, rotation_motor, x_motor=None, z_motor=None,
                        get_ellipse_points=find_needle_tips, num_frames=10, metric_eps=None,
                        position_eps=0.1 * q.deg, max_iterations=5,
                        initial_x_coeff=1 * q.dimensionless, initial_z_coeff=1 * q.dimensionless,
                        shutter=None, flat_motor=None, flat_position=None, y_0=0, y_1=None,
                        get_ellipse_points_kwargs=None, frame_consumer=None):
    """
    align_rotation_axis(camera, rotation_motor, x_motor=None, z_motor=None,
    get_ellipse_points=find_needle_tips, num_frames=10, metric_eps=None,
    position_eps=0.1 * q.deg, max_iterations=5, initial_x_coeff=1 * q.dimensionless,
    initial_z_coeff=1 * q.dimensionless, shutter=None, flat_motor=None, flat_position=None,
    y_0=0, y_1=None, get_ellipse_points_kwargs=None, frame_consumer=None)

    Align rotation axis. *camera* is used to obtain frames, *rotation_motor* rotates the sample
    around the tomographic axis of rotation, *x_motor* turns the sample around x-axis, *z_motor*
    turns the sample around z-axis.

    *get_ellipse_points* is a function with one positional argument, a set of images. It computes
    the ellipse points from the sample positions as it rotates around the tomographic axis.  You can
    use e.g. :func:`concert.imageprocessing.find_needle_tips` and
    :func:`concert.imageprocessing.find_sphere_centers` to extract the ellipse points from needle
    tips or sphere centers. You can pass additional keyword arguments to the *get_ellipse_points*
    function in the *get_ellipse_points_kwargs* dictionary.

    *num_frames* defines how many frames are acquired and passed to the *measure*. *metric_eps* is
    the metric threshold for stopping the procedure. If not specified, it is calculated
    automatically to not exceed 0.5 pixels vertically. If *max_iterations* is reached the procedure
    stops as well. *initial_[x|z]_coeff* is the coefficient applied` to the motor motion for the
    first iteration. If we move the camera instead of the rotation stage, it is often necessary to
    acquire fresh flat fields. In order to make an up-to-date flat correction, specify *shutter* if
    you want fresh dark fields and specify *flat_motor* and *flat_position* to acquire flat fields.
    Crop acquired images to *y_0* and *y_1*. *frame_consumer* is a coroutine which will receive the
    frames acquired at different sample positions.

    The procedure finishes when it finds the minimum angle between an ellipse extracted from the
    sample movement and respective axes or the found angle drops below *metric_eps*. The axis of
    rotation after the procedure is (0,1,0), which is the direction perpendicular to the beam
    direction and the lateral direction. *x_motor* and *z_motor* do not have to move exactly by the
    computed angles but their relative motion must be linear with respect to computed angles (e.g.
    if the motors operate with steps it is fine, also rotation direction does not need to be known).
    """
    if get_ellipse_points_kwargs is None:
        get_ellipse_points_kwargs = {}

    if not x_motor and not z_motor:
        raise ValueError("At least one of the x, z motors must be given")

    def make_step(i, motor, position_last, angle_last, angle_current, initial_coeff,
                  rotation_type):
        LOG.debug("%s: i: %d, last angle: %s, angle: %s, last position: %s, position: %s",
                  rotation_type, i, angle_last.to(q.deg), angle_current.to(q.deg),
                  position_last.to(q.deg), motor.position.to(q.deg))
        if i > 0:
            # Assume linear mapping between the computed angles and motor motion
            if angle_current == angle_last:
                coeff = 0 * q.dimensionless
            else:
                coeff = (motor.position - position_last) / (angle_current - angle_last)
        else:
            coeff = initial_coeff
        position_last = motor.position
        angle_last = angle_current

        # Move relative, i.e. if *angle_current* should go to 0, then we need to move in the
        # other direction with *coeff* applied
        LOG.debug("%s coeff: %s, Next position: %s", rotation_type, coeff.to_base_units(),
                  (motor.position - coeff * angle_current).to(q.deg))
        future = motor.move(-coeff * angle_current)

        return (future, position_last, angle_last)

    def go_to_best_index(motor, history):
        positions, angles = list(zip(*history))
        best_index = np.argmin(np.abs([angle.to_base_units().magnitude for angle in angles]))
        LOG.debug("Best iteration: %d, position: %s, angle: %s",
                  best_index, positions[best_index].to(q.deg), angles[best_index].to(q.deg))
        motor.position = positions[best_index]

    roll_history = []
    pitch_history = []
    center = None

    if z_motor:
        roll_angle_last = 0 * q.deg
        roll_position_last = z_motor.position
        roll_continue = True
    if x_motor:
        pitch_angle_last = 0 * q.deg
        pitch_position_last = x_motor.position
        pitch_continue = True

    for i in range(max_iterations):
        frames = acquire_frames_360(camera, rotation_motor, num_frames, shutter=shutter,
                                    flat_motor=flat_motor, flat_position=flat_position,
                                    y_0=y_0, y_1=y_1, frame_consumer=frame_consumer).result()
        tips = get_ellipse_points(frames, **get_ellipse_points_kwargs)
        roll_angle_current, pitch_angle_current, center = rotation_axis(tips)
        futures = []
        if metric_eps is None:
            metric_eps = np.rad2deg(np.arctan(1 / frames[0].shape[1])) * q.deg
            LOG.debug('Automatically computed metric epsilon: %s', metric_eps)

        if z_motor and roll_continue:
            roll_history.append((z_motor.position, roll_angle_current))
            if (np.abs(roll_angle_current) >= metric_eps and
                    (np.abs(roll_position_last - z_motor.position) >= position_eps or i == 0)):
                roll_res = make_step(i, z_motor, roll_position_last, roll_angle_last,
                                     roll_angle_current, initial_z_coeff, 'roll')
                roll_future, roll_position_last, roll_angle_last = roll_res
                futures.append(roll_future)
            else:
                LOG.debug("Roll epsilon reached")
                roll_continue = False
        if x_motor and pitch_continue:
            pitch_history.append((x_motor.position, pitch_angle_current))
            if (np.abs(pitch_angle_current) >= metric_eps and
                    (np.abs(pitch_position_last - x_motor.position) >= position_eps or i == 0)):
                pitch_res = make_step(i, x_motor, pitch_position_last, pitch_angle_last,
                                      pitch_angle_current, initial_x_coeff, 'pitch')
                pitch_future, pitch_position_last, pitch_angle_last = pitch_res
                futures.append(pitch_future)
            else:
                LOG.debug("Pitch epsilon reached")
                pitch_continue = False

        if not futures:
            # If there are no futures the motors have reached positions at which the computed
            # angles are below threshold
            break

        wait(futures)

    if i == max_iterations - 1:
        LOG.info('Maximum iterations reached')

    if z_motor:
        go_to_best_index(z_motor, roll_history)
    if x_motor:
        go_to_best_index(x_motor, pitch_history)

    return (roll_history, pitch_history, center)


class ProcessException(Exception):

    """
    Exception raised by a process when something goes wrong with the procedure
    it tries to accomplish, e.g. cannot focus, cannot align rotation axis, etc.

    """

    pass
