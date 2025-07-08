import asyncio
import inspect
import time
from itertools import product
from functools import reduce
import logging
from typing import AsyncIterator, Awaitable, List, Optional, Tuple, Dict
import numpy as np
from concert.base import SoftLimitError, AsyncObject
from concert.coroutines.base import background, broadcast
from concert.coroutines.sinks import Result
from concert.quantities import q, Quantity, has_unit
from concert.measures import rotation_axis
from concert.optimization import halver, optimize_parameter
from concert.imageprocessing import flat_correct, find_needle_tips, find_sphere_centers_corr
from concert.helpers import expects, is_iterable, Numeric
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.devices.shutters.base import Shutter
from concert.devices.cameras.base import Camera
from concert.progressbar import wrap_iterable
from concert.typing import ArrayLike


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


####################################################################################################
# Alignment routines to test during beamtime
####################################################################################################
class MotorState(AsyncObject):
    """Provides a compact view of a given rotation motor's state for alignment"""

    _type: str
    _name: str
    _pose: Quantity
    _state: str
    _lower_limit: Quantity
    _upper_limit: Quantity

    async def __ainit__(self, rot_type: str, rot_motor: RotationMotor) -> None:
        self._type = rot_type
        self._name = rot_motor.__class__.__name__
        self._pose = await rot_motor.get_position()
        self._state = await rot_motor.get_state()
        self._lower_limit = await rot_motor["position"].get_lower()
        self._upper_limit = await rot_motor["position"].get_upper()

    def __str__(self) -> str:
        components = [f"{self._type.capitalize()} Motor: {self._name}"]
        components.append(f"- current-position: {self._pose}")
        components.append(f"- state: {self._state}")
        components.append(f"- lower-limit: {self._lower_limit}")
        components.append(f"- upper-limit: {self._upper_limit}")
        return "\n".join(components)


async def set_soft_limits(motor: RotationMotor, limits: Quantity) -> None:
        """Sets soft `limits` for the specified `motor`"""
        await motor["position"].set_lower(limits[0])
        await motor["position"].set_upper(limits[1])

# async def can_move(motor: RotationMotor, relative_move: Quantity) -> bool:
#     """Can the motor move relatively given that the soft-limits are set"""
#     lower_limit: Quantity = await motor["position"].get_lower()
#     upper_limit: Quantity = await motor["position"].get_upper()
#     current: Quantity = await motor.get_position()
#     return (current + relative_move) >= lower_limit and (current + relative_move) <= upper_limit

async def make_step_dynamic(iteration: int,
                    motor: RotationMotor,
                    pose_last: Quantity,
                    angle_last: Quantity,
                    angle_current: Quantity,
                    initial_gain: float,
                    rotation_type: str,
                    logger: logging.Logger,
                    eps_angle_diff: Quantity = 1e-7 * q.deg,
                    ceil_gain: Quantity = 1e3 * q.dimensionless,
                    ceil_rel_move: Quantity = 5 * q.deg) -> Tuple[Quantity, Quantity]:
    """
    Makes a single iteration with dynamically gained proportional controller. Linear mapping between
    motor motion and calculated angle is assumed. In this routine we compute a reverse gain
    dynamically and move relative, i.e. if *angle_current* should go to 0, then we need to move in
    the other direction with gain applied. If *angle_current* is +ve we need to move relative with a
    negative angle and if *angle_current* is -ve we need to move relative with a positive angle to
    proceed towards 0. At any given iteration *angle_current* is the error that we try to minimize.
    Towards the end of the alignment (angle_current - angle_last) may tend to 0 and therefore gain
    may explode. In that situation we scale the gain proportional to the distance to soft-limits.
    """
    func_name: str = inspect.currentframe().f_code.co_name
    pose_current = await motor.get_position()
    iteration_log = f"{func_name}: {rotation_type}: iter: {iteration}"
    logger.info(iteration_log)
    logger.info("=" * len(iteration_log))
    logger.info("last angle: {}, current angle: {}, last position: {}, current position: {}".format(
        angle_last.to(q.deg), angle_current.to(q.deg), pose_last.to(q.deg), pose_current.to(q.deg)))
    angle_diff = angle_current - angle_last
    pose_diff = pose_current - pose_last
    # If we are on the first iteration we use the initial gain to kick start the alignment, else
    # we try to compute the gain from position and angular differences.
    if iteration > 0:
        # If the angle change between previous and current iteration is significant then we compute
        # the dynamic gain as (position change / unit angle change) capped by `ceil_gain`
        # to avoid gigantic move.
        if abs(angle_diff) > eps_angle_diff:
            gain = np.clip(pose_diff / angle_diff, -ceil_gain, ceil_gain)
        # If the angle change is insignificant then we have two possibilities to consider.
        else:
            # If there is a non-zero change in position change but the angle did not change
            # significantly this can mean motor sensitivity issue or a potential non-linear mapping
            # between motor motion and angle change. In this case we calculate a gain with a ceiling
            # value to prevent a explosive motion.
            if abs(pose_diff) > 0:
                gain = ceil_gain * np.sign(pose_diff)
            # If neither angle not position changed significantly we have no gain. Ideally this
            # should mark the end of alignment.
            else:
                gain = 0 * q.dimensionless
    else:
        gain = initial_gain
    pose_last = pose_current
    angle_last = angle_current
    # Relative movement is capped by `ceil_rel_move`.
    move_relative: Quantity = np.clip(-gain * angle_current, -ceil_rel_move, ceil_rel_move)
    logger.info(f"gain: {gain} move-relative: {move_relative}")
    if np.any(np.sign(move_relative.magnitude)):
        try:
            await motor.move(move_relative)
        except SoftLimitError:
            logger.info(f"motor: {rotation_type} encountered soft-limit error")
            logger.info(f"motor: {rotation_type} current-position: {pose_last} current-angle: {angle_last}")
            logger.info("=" * len(iteration_log))
            return pose_last, angle_last
    else:
        logger.info(f"motor: {rotation_type} didn't need to move")
        logger.info(f"motor: {rotation_type} current-position: {pose_last} current-angle: {angle_last}")
        logger.info("=" * len(iteration_log))
        return pose_last, angle_last
    pose_current = await motor.get_position()
    logger.info(f"motor: {rotation_type} moved, current-position: {pose_current} current-angle: {angle_last}")
    logger.info("=" * len(iteration_log))
    return pose_last, angle_last


async def go_to_best_index(motor: RotationMotor, rot_type: str, history: List[Dict[str, Quantity]],
                           logger: logging.Logger) -> None:
    """Sets the motor position against the lowest angular error"""
    func_name: str = inspect.currentframe().f_code.co_name
    positions, angles = list(zip(*[(item["position"], item[rot_type]) for item in history]))
    best_index = np.argmin(np.abs([angle.to_base_units().magnitude for angle in angles]))
    logger.info("{}: best: {}, position: {}, angle: {}".format(func_name, rot_type,
                                                               positions[best_index].to(q.deg),
                                                               angles[best_index].to(q.deg)))
    await motor.set_position(positions[best_index])


async def extract_ellipse_points(producer: AsyncIterator[ArrayLike], radius: int) -> List[ArrayLike]:
    """Finds sphere centers from incoming frames using correlation"""
    return await find_sphere_centers_corr(producer, radius=radius)


@background
async def align_pitch_with_dynamic_gain(camera: Camera,
                                  angular_rot_motor: RotationMotor,
                                  angular_rot_limits: Quantity,
                                  x_rot_motor: RotationMotor,
                                  x_rot_limits: Quantity,
                                  flat_motor: LinearMotor,
                                  shutter: Shutter,
                                  flat_position: Quantity,
                                  radius: int,
                                  crop_y_start: int,
                                  crop_y_end: int,
                                  logger: logging.Logger,
                                  base_log_path: str,
                                  num_frames: int = 40,
                                  max_iterations: int = 10,
                                  initial_x_gain: Quantity = 1 * q.dimensionless,
                                  eps_position: Quantity = 0.1 * q.deg,
                                  eps_metric: Optional[Quantity] = None,
                                  eps_angle_diff: Quantity = 1e-7 * q.deg,
                                  ceil_gain: Quantity = 1e3 * q.dimensionless,
                                  ceil_rel_move: Quantity = 5 * q.deg) -> None:
    """
    Implements dynamically gained alignment routine for pitch angle only. This is a proof of concept
    and easy to debug routine since we are focusing on one specific error. Pitch angle error is also
    more problematic compared to the roll angle error because it is harder to compensate. In presence
    of a roll angle error (rotation w.r.t. the axis along to the beam direction) we still can
    translate the projection by rotating proportionally to the roll angle error and recover to some
    extent. However, pitch angle error takes place due to rotation w.r.t. the coplanar orthogonal
    axis to the beam direction. We cannot recover from this error in a straight forward manner.
    Hence, we put more emphasis on the pitch angle error.
    """
    assert (camera and shutter)
    assert (angular_rot_motor and has_unit(angular_rot_limits, "degree") and len(angular_rot_limits) == 2)
    assert (x_rot_motor and has_unit(x_rot_limits, "degree") and len(x_rot_limits) == 2)
    assert (flat_motor and flat_position and has_unit(flat_position, "millimeter"))
    assert (radius and crop_y_start and crop_y_end)
    func_name: str = inspect.currentframe().f_code.co_name
    log_file = f"{func_name}_{time.ctime().replace(' ', '_').lower()}.log"
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.FileHandler("{0}/{1}.log".format(base_log_path, log_file))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(f"Start: {func_name}")
    logger.info("#" * len(f"Start: {func_name}"))
    # Set soft limits to the motors for safe-alignment.
    try:
        await set_soft_limits(angular_rot_motor, angular_rot_limits)
        await set_soft_limits(x_rot_motor, x_rot_limits)
    except Exception as e:
        logger.info(f"{func_name}: error setting soft-limits, \n{str(e)}")
        logger.info(f"End: {func_name}")
        logger.info("#" * len(f"End: {func_name}"))
        logger.removeHandler(handler)
        return
    # Log initial state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    # Kick-start alignment with initial values.
    pitch_angle_last: Quantity = 0 * q.deg
    pitch_position_last: Quantity = await x_rot_motor.get_position()
    pitch_can_continue = True
    pitch_align_history: List[Dict[str, Quantity]] = []
    frames_result = Result()
    for iteration in range(max_iterations):
        acq_consumers = [extract_ellipse_points, frames_result]
        tips_start = time.perf_counter()
        # Acquire frames from camera and perform flat-field correction.
        frame_producer = acquire_frames_360(camera, angular_rot_motor, num_frames, shutter=shutter,
                                            flat_motor=flat_motor, flat_position=flat_position,
                                            y_0=crop_y_start, y_1=crop_y_end)
        coros = broadcast(frame_producer, *acq_consumers)
        try:
            tips = (await asyncio.gather(*coros))[1]
        except Exception as tips_exc:
            logger.info(f"{func_name}: iteration: {iteration} - error finding reference points: {tips_exc}")
            logger.info("motor State: ====>")
            logger.info(str(await MotorState("angular", angular_rot_motor)))
            logger.info(str(await MotorState("pitch", x_rot_motor)))
            logger.info(f"End: {func_name}")
            logger.info("#" * len(f"End: {func_name}"))
            logger.removeHandler(handler)
            return
        logger.info(f"{func_name}: found {len(tips)} points in {time.perf_counter() - tips_start} seconds.")
        _, pitch_angle_current, center = rotation_axis(tips)
        # Determine an alignment metric epsilon if not provided already.
        if eps_metric is None:
            eps_metric = np.rad2deg(np.arctan(1 / frames_result.result.shape[1])) * q.deg
            logger.info(f"{func_name}: computed metric epsilon: {eps_metric}")
        # Start aligning pitch-angle
        if pitch_can_continue:
            x_rot_pos = await x_rot_motor.get_position()
            pitch_align_history.append({"position": x_rot_pos, "pitch": pitch_angle_current})
            if abs(pitch_angle_current) >= eps_metric \
                    and (abs(pitch_position_last - x_rot_pos) >= eps_position or iteration == 0):
                pitch_position_last, pitch_angle_last = await make_step_dynamic(
                    iteration=iteration, motor=x_rot_motor, pose_last=pitch_position_last,
                    angle_last=pitch_angle_last, angle_current=pitch_angle_current,
                    initial_gain=initial_x_gain, rotation_type="pitch", logger=logger,
                    eps_angle_diff=eps_angle_diff, ceil_gain=ceil_gain,
                    ceil_rel_move=ceil_rel_move)
            else:
                logger.info(f"{func_name}: desired pitch angle threshold reached")
                pitch_can_continue = False
    if iteration == max_iterations - 1:
        logger.info(f"{func_name}: maximum iterations reached")
    # Move to the best known position
    coros = []
    coros.append(go_to_best_index(motor=x_rot_motor, rot_type="pitch", history=pitch_align_history,
                                  logger=logger))
    await asyncio.gather(*coros)
    # Leave the system in stable state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    logger.info(f"End: {func_name}")
    logger.info("#" * len(f"End: {func_name}"))
    logger.removeHandler(handler)


@background
async def align_sequential_with_dynamic_gain(camera: Camera,
                                  angular_rot_motor: RotationMotor,
                                  angular_rot_limits: Quantity,
                                  x_rot_motor: RotationMotor,
                                  x_rot_limits: Quantity,
                                  z_rot_motor: RotationMotor,
                                  z_rot_limits: Quantity,
                                  flat_motor: LinearMotor,
                                  shutter: Shutter,
                                  flat_position: Quantity,
                                  radius: int,
                                  crop_y_start: int,
                                  crop_y_end: int,
                                  logger: logging.Logger,
                                  base_log_path: str,
                                  num_frames: int = 40,
                                  max_iterations: int = 10,
                                  initial_x_gain: Quantity = 1 * q.dimensionless,
                                  initial_z_gain: Quantity = 1 * q.dimensionless,
                                  eps_position: Quantity = 0.1 * q.deg,
                                  eps_metric: Optional[Quantity] = None,
                                  eps_angle_diff: Quantity = 1e-7 * q.deg,
                                  ceil_gain: Quantity = 1e3 * q.dimensionless,
                                  ceil_rel_move: Quantity = 5 * q.deg) -> None:
    """
    Implements dynamically gained iterative alignment routine for pitch angle and roll angle
    correction. In each iteration we make a step for roll angle correction followed by a step for
    pitch angle correction.
    """
    assert (camera and shutter)
    assert (angular_rot_motor and has_unit(angular_rot_limits, "degree") and len(angular_rot_limits) == 2)
    assert (x_rot_motor and has_unit(x_rot_limits, "degree") and len(x_rot_limits) == 2)
    assert (z_rot_motor and has_unit(z_rot_limits, "degree") and len(z_rot_limits) == 2)
    assert (flat_motor and flat_position and has_unit(flat_position, "millimeter"))
    assert (radius and crop_y_start and crop_y_end)
    func_name: str = inspect.currentframe().f_code.co_name
    log_file = f"{func_name}_{time.ctime().replace(' ', '_').lower()}.log"
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.FileHandler("{0}/{1}.log".format(base_log_path, log_file))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(f"Start: {func_name}")
    logger.info("#" * len(f"Start: {func_name}"))
    # Set soft limits to the motors for safe-alignment.
    try:
        await set_soft_limits(angular_rot_motor, angular_rot_limits)
        await set_soft_limits(x_rot_motor, x_rot_limits)
        await set_soft_limits(z_rot_motor, z_rot_limits)
    except Exception as e:
        logger.info(f"{func_name}: error setting soft-limits, \n{str(e)}")
        logger.info(f"End: {func_name}")
        logger.info("#" * len(f"End: {func_name}"))
        logger.removeHandler(handler)
        return
    # Log initial state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("roll", z_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    # Kick-start alignment with initial values.
    if z_rot_motor:
        roll_angle_last: Quantity = 0 * q.deg
        roll_position_last: Quantity = await z_rot_motor.get_position()
        roll_can_continue = True
        roll_align_history: List[Dict[str, Quantity]] = []
    if x_rot_motor:
        pitch_angle_last: Quantity = 0 * q.deg
        pitch_position_last: Quantity = await x_rot_motor.get_position()
        pitch_can_continue = True
        pitch_align_history: List[Dict[str, Quantity]] = []
    frames_result = Result()
    for iteration in range(max_iterations):
        acq_consumers = [extract_ellipse_points, frames_result]
        tips_start = time.perf_counter()
        # Acquire frames from camera and perform flat-field correction.
        frame_producer = acquire_frames_360(camera, angular_rot_motor, num_frames, shutter=shutter,
                                            flat_motor=flat_motor, flat_position=flat_position,
                                            y_0=crop_y_start, y_1=crop_y_end)
        coros = broadcast(frame_producer, *acq_consumers)
        try:
            tips = (await asyncio.gather(*coros))[1]
        except Exception as tips_exc:
            logger.info(f"{func_name}: iteration: {iteration} - error finding reference points: {tips_exc}")
            logger.info("motor State: ====>")
            logger.info(str(await MotorState("angular", angular_rot_motor)))
            logger.info(str(await MotorState("roll", z_rot_motor)))
            logger.info(str(await MotorState("pitch", x_rot_motor)))
            logger.info(f"End: {func_name}")
            logger.info("#" * len(f"End: {func_name}"))
            logger.removeHandler(handler)
            return
        logger.info(f"{func_name}: found {len(tips)} points in {time.perf_counter() - tips_start} seconds.")
        roll_angle_current, pitch_angle_current, center = rotation_axis(tips)
        # Determine an alignment metric epsilon if not provided already.
        if eps_metric is None:
            eps_metric = np.rad2deg(np.arctan(1 / frames_result.result.shape[1])) * q.deg
            logger.info(f"{func_name}: computed metric epsilon: {eps_metric}")
        # Start aligning roll-angle
        if roll_can_continue:
            z_rot_pos = await z_rot_motor.get_position()
            roll_align_history.append({"position": z_rot_pos, "roll": roll_angle_current})
            if abs(roll_angle_current) >= eps_metric \
                and (abs(roll_position_last - z_rot_pos) >= eps_position or iteration == 0):
                roll_position_last, roll_angle_last = await make_step_dynamic(
                    iteration=iteration, motor=z_rot_motor, pose_last=roll_position_last,
                    angle_last=roll_angle_last, angle_current=roll_angle_current,
                    initial_gain=initial_z_gain, rotation_type="roll", logger=logger,
                    eps_angle_diff=eps_angle_diff, ceil_gain=ceil_gain,
                    ceil_rel_move=ceil_rel_move)
            else:
                logger.info(f"{func_name}: desired roll angle threshold reached")
                roll_can_continue = False
        # Start aligning pitch-angle
        if pitch_can_continue:
            x_rot_pos = await x_rot_motor.get_position()
            pitch_align_history.append({"position": x_rot_pos, "pitch": pitch_angle_current})
            if abs(pitch_angle_current) >= eps_metric \
                    and (abs(pitch_position_last - x_rot_pos) >= eps_position or iteration == 0):
                pitch_position_last, pitch_angle_last = await make_step_dynamic(
                    iteration=iteration, motor=x_rot_motor, pose_last=pitch_position_last,
                    angle_last=pitch_angle_last, angle_current=pitch_angle_current,
                    initial_gain=initial_x_gain, rotation_type="pitch", logger=logger,
                    eps_angle_diff=eps_angle_diff, ceil_gain=ceil_gain,
                    ceil_rel_move=ceil_rel_move)
            else:
                logger.info(f"{func_name}: desired pitch angle threshold reached")
                pitch_can_continue = False
    if iteration == max_iterations - 1:
        logger.info(f"{func_name}: maximum iterations reached")
    # Move to the best known position
    coros = []
    coros.append(go_to_best_index(motor=z_rot_motor, rot_type="roll", history=roll_align_history,
                                  logger=logger))
    coros.append(go_to_best_index(motor=x_rot_motor, rot_type="pitch", history=pitch_align_history,
                                  logger=logger))
    await asyncio.gather(*coros)
    # Leave the system in stable state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("roll", z_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    logger.info(f"End: {func_name}")
    logger.info("#" * len(f"End: {func_name}"))
    logger.removeHandler(handler)


async def make_step_static(iteration: int,
                    motor: RotationMotor,
                    current_angle: Quantity,
                    rotation_type: str,
                    logger: logging.Logger,
                    error_state: Dict[str, float],
                    proportional_gain: float = 50.0,
                    integral_gain: float = 0.5,
                    derivative_gain: float = 0.0,
                    ceil_rel_move: Quantity = 5 * q.deg) -> None:
    """
    Makes a single step using proportional integral and optionally derivative gain controller.
    """
    func_name: str = inspect.currentframe().f_code.co_name
    pose_current = await motor.get_position()
    iteration_log = f"{func_name}: {rotation_type}: iter: {iteration}"
    logger.info(iteration_log)
    logger.info("=" * len(iteration_log))
    ceil_integral: float = ceil_rel_move.magnitude / integral_gain
    target_angle: float = 0.0
    current_error: float = target_angle - current_angle.magnitude
    logger.info(f"{func_name}: current {rotation_type} error: {current_error * q.deg}")
    # Propotional Correction
    _proportional = proportional_gain * current_error
    # Integral Corrrection
    if rotation_type == "roll":
        error_state["accumulated_roll_error"] += current_error
        error_state["accumulated_roll_error"] = np.clip(error_state["accumulated_roll_error"],
                                                        -ceil_integral, ceil_integral)
        _integral = integral_gain * error_state["accumulated_roll_error"]
    elif rotation_type == "pitch":
        error_state["accumulated_pitch_error"] += current_error
        error_state["accumulated_pitch_error"] = np.clip(error_state["accumulated_pitch_error"],
                                                        -ceil_integral, ceil_integral)
        _integral = integral_gain * error_state["accumulated_pitch_error"]
    # Derivative Correction
    if rotation_type == "roll":
        derivative_error = current_error - error_state["last_roll_error"]
    elif rotation_type == "pitch":
        derivative_error = current_error - error_state["last_pitch_error"]
    _derivative = derivative_gain * derivative_error
    move_raw: float = _proportional + _integral + _derivative
    move_relative: float = np.clip(move_raw, -ceil_rel_move, ceil_rel_move)
    # Update state for next iteration
    if rotation_type == "roll":
        error_state["last_roll_error"] = current_error
    elif rotation_type == "pitch":
        error_state["last_pitch_error"] = current_error
    # We check for a valid movement before asking the motor to move.
    logger.info(f"{func_name}: current {rotation_type} move-raw: {move_raw} move-clipped: {move_relative}")
    if np.any(np.sign(move_relative)):
        try:
            await motor.move(move_relative * q.deg)
        except SoftLimitError:
            logger.info(f"{func_name}: motor: {rotation_type} encountered soft-limit error")
    else:
        logger.info(f"{func_name}: motor: {rotation_type} didn't need to move")
    logger.info(f"{func_name}: motor {rotation_type} moved to position: {await motor.get_position()}")
    logger.info("=" * len(iteration_log))


@background
async def align_pitch_with_static_gain(camera: Camera,
                                  angular_rot_motor: RotationMotor,
                                  angular_rot_limits: Quantity,
                                  x_rot_motor: RotationMotor,
                                  x_rot_limits: Quantity,
                                  flat_motor: LinearMotor,
                                  shutter: Shutter,
                                  flat_position: Quantity,
                                  radius: int,
                                  crop_y_start: int,
                                  crop_y_end: int,
                                  logger: logging.Logger,
                                  base_log_path: str,
                                  num_frames: int = 40,
                                  max_iterations: int = 20,
                                  proportional_gain: float = 50.0,
                                  integral_gain: float = 0.5,
                                  derivative_gain: float = 0.0,
                                  eps_metric: Quantity = 1e-5 * q.deg,
                                  ceil_rel_move: Quantity = 5 * q.deg) -> None:
    """
    Implements statically gained alignment routine for pitch angle only. This is a proof of concept
    and easy to debug routine since we are focusing on one specific error. Pitch angle error is also
    more problematic compared to the roll angle error because it is harder to compensate. In presence
    of a roll angle error (rotation w.r.t. the axis along to the beam direction) we still can
    translate the projection by rotating proportionally to the roll angle error and recover to some
    extent. However, pitch angle error takes place due to rotation w.r.t. the coplanar orthogonal
    axis to the beam direction. We cannot recover from this error in a straight forward manner.
    Hence, we put more emphasis on the pitch angle error.
    """
    assert (camera and shutter)
    assert (angular_rot_motor and has_unit(angular_rot_limits, "degree") and len(angular_rot_limits) == 2)
    assert (x_rot_motor and has_unit(x_rot_limits, "degree") and len(x_rot_limits) == 2)
    assert (flat_motor and flat_position and has_unit(flat_position, "millimeter"))
    assert (radius and crop_y_start and crop_y_end)
    func_name: str = inspect.currentframe().f_code.co_name
    log_file = f"{func_name}_{time.ctime().replace(' ', '_').lower()}.log"
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.FileHandler("{0}/{1}.log".format(base_log_path, log_file))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(f"Start: {func_name}")
    logger.info("#" * len(f"Start: {func_name}"))
    # Set soft limits to the motors for safe-alignment.
    try:
        await set_soft_limits(angular_rot_motor, angular_rot_limits)
        await set_soft_limits(x_rot_motor, x_rot_limits)
    except Exception as e:
        logger.info(f"{func_name}: error setting soft-limits, \n{str(e)}")
        logger.info(f"End: {func_name}")
        logger.info("#" * len(f"End: {func_name}"))
        logger.removeHandler(handler)
        return
    # Log initial state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    # Kick-start alignment with initial values.
    error_state: Dict[str, float] = {
        "last_roll_error": 0.0,
        "accumulated_roll_error": 0.0,
        "last_pitch_error": 0.0,
        "accumulated_pitch_error": 0.0
    }
    if x_rot_motor:
        pitch_can_continue = True
        pitch_align_history: List[Dict[str, Quantity]] = []
    frames_result = Result()
    for iteration in range(max_iterations):
        acq_consumers = [extract_ellipse_points, frames_result]
        tips_start = time.perf_counter()
        # Acquire frames from camera and perform flat-field correction.
        frame_producer = acquire_frames_360(camera, angular_rot_motor, num_frames, shutter=shutter,
                                            flat_motor=flat_motor, flat_position=flat_position,
                                            y_0=crop_y_start, y_1=crop_y_end)
        coros = broadcast(frame_producer, *acq_consumers)
        try:
            tips = (await asyncio.gather(*coros))[1]
        except Exception as tips_exc:
            logger.info(f"{func_name}: iteration: {iteration} - error finding reference points: {tips_exc}")
            logger.info("motor State: ====>")
            logger.info(str(await MotorState("angular", angular_rot_motor)))
            logger.info(str(await MotorState("pitch", x_rot_motor)))
            logger.info(f"End: {func_name}")
            logger.info("#" * len(f"End: {func_name}"))
            logger.removeHandler(handler)
            return
        logger.info(f"{func_name}: found {len(tips)} points in {time.perf_counter() - tips_start} seconds.")
        roll_angle_current, pitch_angle_current, center = rotation_axis(tips)
        # Determine an alignment metric epsilon if not provided already.
        if eps_metric is None:
            eps_metric = np.rad2deg(np.arctan(1 / frames_result.result.shape[1])) * q.deg
            logger.info(f"{func_name}: computed metric epsilon: {eps_metric}")
        # Start aligning pitch-angle
        if pitch_can_continue:
            if abs(pitch_angle_current) >= eps_metric:
                pitch_align_history.append({"position": await x_rot_motor.get_position(),
                                            "pitch": pitch_angle_current})
                await make_step_static(iteration=iteration, motor=x_rot_motor,
                                       current_angle=pitch_angle_current, rotation_type="pitch",
                                       logger=logger, error_state=error_state,
                                       proportional_gain=proportional_gain, integral_gain=integral_gain,
                                       derivative_gain=derivative_gain, ceil_rel_move=ceil_rel_move)
            else:
                logger.info(f"{func_name}: desired pitch angle threshold reached")
                pitch_can_continue = False
    if iteration == max_iterations - 1:
        logger.info(f"{func_name}: maximum iterations reached")
    # Move to the best known position
    coros = []
    coros.append(go_to_best_index(motor=x_rot_motor, rot_type="pitch", history=pitch_align_history,
                                  logger=logger))
    await asyncio.gather(*coros)
    # Leave the system in stable state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    logger.info(f"End: {func_name}")
    logger.info("#" * len(f"End: {func_name}"))
    logger.removeHandler(handler)


@background
async def align_sequential_with_static_gain(camera: Camera,
                                  angular_rot_motor: RotationMotor,
                                  angular_rot_limits: Quantity,
                                  x_rot_motor: RotationMotor,
                                  x_rot_limits: Quantity,
                                  z_rot_motor: RotationMotor,
                                  z_rot_limits: Quantity,
                                  flat_motor: LinearMotor,
                                  shutter: Shutter,
                                  flat_position: Quantity,
                                  radius: int,
                                  crop_y_start: int,
                                  crop_y_end: int,
                                  logger: logging.Logger,
                                  base_log_path: str,
                                  num_frames: int = 40,
                                  max_iterations: int = 20,
                                  proportional_gain: float = 50.0,
                                  integral_gain: float = 0.5,
                                  derivative_gain: float = 0.0,
                                  eps_metric: Quantity = 1e-5 * q.deg,
                                  ceil_rel_move: Quantity = 5 * q.deg) -> None:
    """
    Implements statically gained iterative alignment routine for pitch angle and roll angle
    correction. In each iteration we make a step for roll angle correction followed by a step for
    pitch angle correction.
    """
    assert (camera and shutter)
    assert (angular_rot_motor and has_unit(angular_rot_limits, "degree") and len(angular_rot_limits) == 2)
    assert (x_rot_motor and has_unit(x_rot_limits, "degree") and len(x_rot_limits) == 2)
    assert (z_rot_motor and has_unit(z_rot_limits, "degree") and len(z_rot_limits) == 2)
    assert (flat_motor and flat_position and has_unit(flat_position, "millimeter"))
    assert (radius and crop_y_start and crop_y_end)
    func_name: str = inspect.currentframe().f_code.co_name
    log_file = f"{func_name}_{time.ctime().replace(' ', '_').lower()}.log"
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.FileHandler("{0}/{1}.log".format(base_log_path, log_file))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(f"Start: {func_name}")
    logger.info("#" * len(f"Start: {func_name}"))
    # Set soft limits to the motors for safe-alignment.
    try:
        await set_soft_limits(angular_rot_motor, angular_rot_limits)
        await set_soft_limits(x_rot_motor, x_rot_limits)
        await set_soft_limits(z_rot_motor, z_rot_limits)
    except Exception as e:
        logger.info(f"{func_name}: error setting soft-limits, \n{str(e)}")
        logger.info(f"End: {func_name}")
        logger.info("#" * len(f"End: {func_name}"))
        logger.removeHandler(handler)
        return
    # Log initial state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("roll", z_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    # Kick-start alignment with initial values.
    error_state: Dict[str, float] = {
        "last_roll_error": 0.0,
        "accumulated_roll_error": 0.0,
        "last_pitch_error": 0.0,
        "accumulated_pitch_error": 0.0
    }
    if z_rot_motor:
        roll_can_continue = True
        roll_align_history: List[Dict[str, Quantity]] = []
    if x_rot_motor:
        pitch_can_continue = True
        pitch_align_history: List[Dict[str, Quantity]] = []
    frames_result = Result()
    for iteration in range(max_iterations):
        acq_consumers = [extract_ellipse_points, frames_result]
        tips_start = time.perf_counter()
        # Acquire frames from camera and perform flat-field correction.
        frame_producer = acquire_frames_360(camera, angular_rot_motor, num_frames, shutter=shutter,
                                            flat_motor=flat_motor, flat_position=flat_position,
                                            y_0=crop_y_start, y_1=crop_y_end)
        coros = broadcast(frame_producer, *acq_consumers)
        try:
            tips = (await asyncio.gather(*coros))[1]
        except Exception as tips_exc:
            logger.info(f"{func_name}: iteration: {iteration} - error finding reference points: {tips_exc}")
            logger.info("motor State: ====>")
            logger.info(str(await MotorState("angular", angular_rot_motor)))
            logger.info(str(await MotorState("roll", z_rot_motor)))
            logger.info(str(await MotorState("pitch", x_rot_motor)))
            logger.info(f"End: {func_name}")
            logger.info("#" * len(f"End: {func_name}"))
            logger.removeHandler(handler)
            return
        logger.info(f"{func_name}: found {len(tips)} points in {time.perf_counter() - tips_start} seconds.")
        roll_angle_current, pitch_angle_current, center = rotation_axis(tips)
        # Determine an alignment metric epsilon if not provided already.
        if eps_metric is None:
            eps_metric = np.rad2deg(np.arctan(1 / frames_result.result.shape[1])) * q.deg
            logger.info(f"{func_name}: computed metric epsilon: {eps_metric}")
        # Start aligning roll-angle
        if roll_can_continue:
            if abs(roll_angle_current) >= eps_metric:
                roll_align_history.append({"position": await z_rot_motor.get_position(),
                                           "roll": roll_angle_current})
                await make_step_static(iteration=iteration, motor=z_rot_motor,
                                       current_angle=roll_angle_current, rotation_type="roll",
                                       logger=logger, error_state=error_state,
                                       proportional_gain=proportional_gain, integral_gain=integral_gain,
                                       derivative_gain=derivative_gain, ceil_rel_move=ceil_rel_move)
            else:
                logger.info(f"{func_name}: desired roll angle threshold reached")
                roll_can_continue = False
        # Start aligning pitch-angle
        if pitch_can_continue:
            if abs(pitch_angle_current) >= eps_metric:
                pitch_align_history.append({"position": await x_rot_motor.get_position(),
                                            "pitch": pitch_angle_current})
                await make_step_static(iteration=iteration, motor=x_rot_motor,
                                       current_angle=pitch_angle_current, rotation_type="pitch",
                                       logger=logger, error_state=error_state,
                                       proportional_gain=proportional_gain, integral_gain=integral_gain,
                                       derivative_gain=derivative_gain, ceil_rel_move=ceil_rel_move)
            else:
                logger.info(f"{func_name}: desired pitch angle threshold reached")
                pitch_can_continue = False
    if iteration == max_iterations - 1:
        logger.info(f"{func_name}: maximum iterations reached")
    # Move to the best known position
    coros = []
    coros.append(go_to_best_index(motor=z_rot_motor, rot_type="roll", history=roll_align_history,
                                  logger=logger))
    coros.append(go_to_best_index(motor=x_rot_motor, rot_type="pitch", history=pitch_align_history,
                                  logger=logger))
    await asyncio.gather(*coros)
    # Leave the system in stable state
    logger.info("motor State: ====>")
    logger.info(str(await MotorState("angular", angular_rot_motor)))
    logger.info(str(await MotorState("roll", z_rot_motor)))
    logger.info(str(await MotorState("pitch", x_rot_motor)))
    logger.info(f"End: {func_name}")
    logger.info("#" * len(f"End: {func_name}"))
    logger.removeHandler(handler)
####################################################################################################


class ProcessError(Exception):

    """
    Exception raised by a process when something goes wrong with the procedure
    it tries to accomplish, e.g. cannot focus, cannot align rotation axis, etc.

    """

    

    pass



