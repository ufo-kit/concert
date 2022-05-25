import asyncio
import time
from itertools import product
from functools import reduce
import numpy as np
import logging
from concert.coroutines.base import background, broadcast
from concert.coroutines.sinks import Result
from concert.quantities import q
from concert.measures import rotation_axis
from concert.optimization import halver, optimize_parameter
from concert.imageprocessing import flat_correct, find_needle_tips
from concert.helpers import expects, is_iterable, Numeric
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.devices.shutters.base import Shutter
from concert.devices.cameras.base import Camera
from concert.progressbar import wrap_iterable


LOG = logging.getLogger(__name__)


async def scan(params, values, feedback, go_back=False):
    """Multi-dimensional scan of :class:`concert.base.Parameter` instances *params*, which are set
    to *values*. *feedback* is a coroutine function without parameters called after every iteration.
    If *go_back* is True, the original parameter values are restored at the end.

    If *params* is just one parameter and *values* is one list of values, perform a 1D scan. In this
    case tuples (x, y) are returned where x are the individual elements from the list of *values*
    and y = feedback() is called after every value setting.

    If *params* is a list of parameters and *values* is a list of lists, assign values[i] to
    params[i] and do a multi-dimensional scan, where last parameter changes the fastest (in other
    words a nested scan of all parameters, where *feedback* is called for every combination of
    parameter values. The combinations are obtained as a cartesian product of the *values*. For
    example, scanning camera exposure times and motor positions with values=[[1, 2] * q.s, [3, 5] *
    q.mm], would result in this::

        [((1 * q.s, 3 * q.mm), feedback()), ((1 * q.s, 5 * q.mm), feedback()),
         ((2 * q.s, 3 * q.mm), feedback()), ((2 * q.s, 5 * q.mm), feedback())]

    In general, for n parameters and lists of values, returned are tuples ((x_0, ..., x_{n-1}), y),
    where y = feedback() is called after every value setting (any parameter change). Parameter
    setting occurs in parallel, is waited for and then *feedback* is called.

    A simple 1D example::

        async for vector in scan(camera['exposure_time'], np.arange(1, 10, 1) * q.s, feedback):
            print(vector) # prints (1 * q.s, feedback()) and so on

    2D example::

        params = [camera['exposure_time'], motor['position']]
        values = [np.arange(1, 10, 1) * q.s, np.arange(5, 15, 2) * q.mm]
        async for vector in scan(params, values, feedback):
            print(vector) # prints ((1 * q.s, 5 * q.mm), feedback()) and so on

    """
    params = params if is_iterable(params) else [params]
    ndim = len(params)
    values = values if is_iterable(values[0]) else [values]

    if ndim > 1 and len(params) != len(values):
        raise RuntimeError
    num_iterations = reduce(lambda x, y: x * y, [len(vals) for vals in values])

    if go_back:
        for param in params:
            await param.stash()

    try:
        for vector in wrap_iterable(product(*values), total=num_iterations):
            setters = [params[i].set(vector[i]) for i in range(len(vector))]
            await asyncio.gather(*setters)
            yield (vector[0] if ndim == 1 else vector, await feedback())
    finally:
        if go_back:
            for param in params:
                await param.restore()


async def ascan(param, start, stop, step, feedback, go_back=False, include_last=True):
    """A convenience function to perform a 1D scan on parameter *param*, scan from *start* value to
    *stop* with *step*. *feedback* and *go_back* are the same as in the :func:`.scan`. If
    *include_last* is True, the *stop* value will be included in the created values This function
    just computes the values from *start*, *stop*, *step* and then calls :func:`.scan`::

        scan(param, values, feedback=feedback, go_back=go_back))
    """
    stop = stop.to(start.units)
    step = step.to(start.units)
    region = np.arange(start.magnitude, stop.magnitude, step.magnitude)
    if include_last:
        region = np.concatenate((region, [stop.magnitude]))

    async for item in scan(param, region * start.units, feedback, go_back=go_back):
        yield item


async def dscan(param, delta, step, feedback, go_back=False, include_last=True):
    """A convenience function to perform a 1D scan on parameter *param*, scan from its current value
    to some *delta* with *step*. *feedback* and *go_back* are the same as in the :func:`.scan`. This
    function just computes the start and stop values and calls :func:`.ascan`::

        start = await param.get()
        ascan(param, start, start + delta, step, feedback, go_back=go_back)
    """
    start = await param.get()

    async for item in ascan(param, start, start + delta, step, feedback, go_back=go_back,
                            include_last=include_last):
        yield item


@background
@expects(Camera, LinearMotor, measure=None, opt_kwargs=None,
         plot_callback=None, frame_callback=None)
async def focus(camera, motor, measure=np.std, opt_kwargs=None,
                plot_callback=None, frame_callback=None):
    """
    Focus *camera* by moving *motor*. *measure* is a callable that computes a scalar that has to be
    maximized from an image taken with *camera*.  *opt_kwargs* are keyword arguments sent to the
    optimization algorithm.  *plot_callback* is (x, y) values, where x is the iteration number and y
    the metric result.  *frame_callback* is a coroutine function fed with the incoming frames.

    This function is returning a future encapsulating the focusing event. Note, that the camera is
    stopped from recording as soon as the optimal position is found.
    """
    x_linear = 0

    if opt_kwargs is None:
        opt_kwargs = {'initial_step': 0.5 * q.mm,
                      'epsilon': 1e-2 * q.mm}

    async def get_measure():
        await camera.trigger()
        frame = await camera.grab()
        if frame_callback:
            await frame_callback(frame)
        return - measure(frame)

    async def linearize_optimization(xy):
        """
        Make the optimization's (x, y) x component monotonically increasing,
        otherwise the the plot update is not lucid.
        """
        nonlocal x_linear
        if plot_callback:
            await plot_callback((x_linear, xy[1]))
        x_linear += 1

    await camera['trigger_source'].stash()
    await camera.set_trigger_source(camera.trigger_sources.SOFTWARE)

    try:
        async with camera.recording():
            await optimize_parameter(motor['position'], get_measure,
                                     await motor.get_position(),
                                     halver, alg_kwargs=opt_kwargs,
                                     callback=linearize_optimization)
    finally:
        await camera['trigger_source'].restore()


@expects(Camera, RotationMotor, num_frames=Numeric(1), shutter=Shutter,
         flat_motor=LinearMotor, flat_position=Numeric(1, q.m), y_0=Numeric(1),
         y_1=Numeric(1))
async def acquire_frames_360(camera, rotation_motor, num_frames, shutter=None, flat_motor=None,
                             flat_position=None, y_0=0, y_1=None):
    """
    acquire_frames_360(camera, rotation_motor, num_frames, shutter=None, flat_motor=None,
                       flat_position=None, y_0=0, y_1=None)

    Acquire frames around a circle.
    """
    flat = None
    if await camera.get_state() == 'recording':
        await camera.stop_recording()
    await camera['trigger_source'].stash()
    await camera.set_trigger_source(camera.trigger_sources.SOFTWARE)

    try:
        async with camera.recording():
            if shutter:
                if await shutter.get_state() != 'closed':
                    await shutter.close()
                await camera.trigger()
                dark = (await camera.grab())[y_0:y_1]
                await shutter.open()
            if flat_motor and flat_position is not None:
                radio_position = await flat_motor.get_position()
                await flat_motor.set_position(flat_position)
                await camera.trigger()
                flat = (await camera.grab())[y_0:y_1]
                await flat_motor.set_position(radio_position)
            for i in range(num_frames):
                await rotation_motor.move(2 * np.pi / num_frames * q.rad)
                await camera.trigger()
                frame = (await camera.grab())[y_0:y_1].astype(float)
                if flat is not None:
                    frame = flat_correct(frame, flat, dark=dark)
                    frame = np.nan_to_num(-np.log(frame))
                    # Huge numbers can also cause trouble
                    frame[np.abs(frame) > 1e6] = 0
                else:
                    frame = frame.max() - frame
                yield frame
    finally:
        # No side effects
        await camera['trigger_source'].restore()


@background
@expects(Camera, RotationMotor, x_motor=RotationMotor, z_motor=RotationMotor,
         get_ellipse_points=find_needle_tips, num_frames=Numeric(1), metric_eps=Numeric(1, q.deg),
         position_eps=Numeric(1, q.deg), max_iterations=Numeric(1),
         initial_x_coeff=Numeric(1, q.dimensionless), initial_z_coeff=Numeric(1, q.dimensionless),
         shutter=Shutter, flat_motor=LinearMotor, flat_position=Numeric(1, q.m),
         y_0=Numeric(1), y_1=Numeric(1), get_ellipse_points_kwargs=None, frame_consumers=None)
async def align_rotation_axis(camera, rotation_motor, x_motor=None, z_motor=None,
                              get_ellipse_points=find_needle_tips, num_frames=10, metric_eps=None,
                              position_eps=0.1 * q.deg, max_iterations=5,
                              initial_x_coeff=1 * q.dimensionless,
                              initial_z_coeff=1 * q.dimensionless,
                              shutter=None, flat_motor=None, flat_position=None, y_0=0, y_1=None,
                              get_ellipse_points_kwargs=None, frame_consumers=None):
    """
    align_rotation_axis(camera, rotation_motor, x_motor=None, z_motor=None,
    get_ellipse_points=find_needle_tips, num_frames=10, metric_eps=None,
    position_eps=0.1 * q.deg, max_iterations=5, initial_x_coeff=1 * q.dimensionless,
    initial_z_coeff=1 * q.dimensionless, shutter=None, flat_motor=None, flat_position=None,
    y_0=0, y_1=None, get_ellipse_points_kwargs=None, frame_consumers=None)

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
    Crop acquired images to *y_0* and *y_1*. *frame_consumers* are coroutine functions which will be
    fed with all acquired frames.

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
        raise ProcessError("At least one of the x, z motors must be given")

    async def make_step(i, motor, position_last, angle_last, angle_current, initial_coeff,
                        rotation_type):
        cur_pos = await motor.get_position()
        LOG.debug("%s: i: %d, last angle: %s, angle: %s, last position: %s, position: %s",
                  rotation_type, i, angle_last.to(q.deg), angle_current.to(q.deg),
                  position_last.to(q.deg), cur_pos.to(q.deg))
        if i > 0:
            # Assume linear mapping between the computed angles and motor motion
            if angle_current == angle_last:
                coeff = 0 * q.dimensionless
            else:
                coeff = (cur_pos - position_last) / (angle_current - angle_last)
        else:
            coeff = initial_coeff
        position_last = cur_pos
        angle_last = angle_current

        # Move relative, i.e. if *angle_current* should go to 0, then we need to move in the
        # other direction with *coeff* applied
        LOG.debug("%s coeff: %s, Next position: %s", rotation_type, coeff.to_base_units(),
                  (cur_pos - coeff * angle_current).to(q.deg))
        await motor.move(-coeff * angle_current)

        return (position_last, angle_last)

    async def go_to_best_index(motor, history):
        positions, angles = list(zip(*history))
        best_index = np.argmin(np.abs([angle.to_base_units().magnitude for angle in angles]))
        LOG.debug("Best iteration: %d, position: %s, angle: %s",
                  best_index, positions[best_index].to(q.deg), angles[best_index].to(q.deg))
        await motor.set_position(positions[best_index])

    async def extract_points(producer):
        return await get_ellipse_points(producer, **get_ellipse_points_kwargs)

    roll_history = []
    pitch_history = []
    center = None

    if z_motor:
        roll_angle_last = 0 * q.deg
        roll_position_last = await z_motor.get_position()
        roll_continue = True
    if x_motor:
        pitch_angle_last = 0 * q.deg
        pitch_position_last = await x_motor.get_position()
        pitch_continue = True

    frames_result = Result()
    for i in range(max_iterations):
        acq_consumers = [extract_points, frames_result]
        if frame_consumers is not None:
            acq_consumers += list(frame_consumers)
        tips_start = time.perf_counter()
        frame_producer = acquire_frames_360(camera, rotation_motor, num_frames, shutter=shutter,
                                            flat_motor=flat_motor, flat_position=flat_position,
                                            y_0=y_0, y_1=y_1)

        coros = broadcast(frame_producer, *acq_consumers)
        try:
            tips = (await asyncio.gather(*coros))[1]
        except Exception as tips_exc:
            raise ProcessError('Error finding reference points') from tips_exc
        LOG.debug('Found %d points in %g s', len(tips), time.perf_counter() - tips_start)
        roll_angle_current, pitch_angle_current, center = rotation_axis(tips)
        coros = []
        x_coro = z_coro = None
        if metric_eps is None:
            metric_eps = np.rad2deg(np.arctan(1 / frames_result.result.shape[1])) * q.deg
            LOG.debug('Automatically computed metric epsilon: %s', metric_eps)

        if z_motor and roll_continue:
            z_pos = await z_motor.get_position()
            roll_history.append((z_pos, roll_angle_current))
            if (np.abs(roll_angle_current) >= metric_eps
                    and (np.abs(roll_position_last - z_pos) >= position_eps or i == 0)):
                z_coro = make_step(i, z_motor, roll_position_last, roll_angle_last,
                                   roll_angle_current, initial_z_coeff, 'roll')
                coros.append(z_coro)
            else:
                LOG.debug("Roll epsilon reached")
                roll_continue = False
        if x_motor and pitch_continue:
            x_pos = await x_motor.get_position()
            pitch_history.append((x_pos, pitch_angle_current))
            if (np.abs(pitch_angle_current) >= metric_eps
                    and (np.abs(pitch_position_last - x_pos) >= position_eps or i == 0)):
                x_coro = make_step(i, x_motor, pitch_position_last, pitch_angle_last,
                                   pitch_angle_current, initial_x_coeff, 'pitch')
                coros.append(x_coro)
            else:
                LOG.debug("Pitch epsilon reached")
                pitch_continue = False

        if not coros:
            # If there are no coros the motors have reached positions at which the computed
            # angles are below threshold
            break

        step_results = await asyncio.gather(*coros)
        if x_coro:
            # Regardless from x_coro and z_coro to be present, x_coro is always added last, so pop
            # it first
            pitch_position_last, pitch_angle_last = step_results.pop()
        if z_coro:
            roll_position_last, roll_angle_last = step_results.pop()

    if i == max_iterations - 1:
        LOG.info('Maximum iterations reached')

    # Move to the best known position
    coros = []
    if z_motor:
        coros.append(go_to_best_index(z_motor, roll_history))
    if x_motor:
        coros.append(go_to_best_index(x_motor, pitch_history))
    await asyncio.gather(*coros)

    return (roll_history, pitch_history, center)


class ProcessError(Exception):

    """
    Exception raised by a process when something goes wrong with the procedure
    it tries to accomplish, e.g. cannot focus, cannot align rotation axis, etc.

    """

    pass
