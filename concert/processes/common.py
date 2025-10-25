import asyncio
from dataclasses import dataclass
import functools
import time
from itertools import product
from functools import reduce
import logging
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np
from concert.base import LimitError, AsyncObject
from concert.coroutines.base import background, broadcast
from concert.coroutines.sinks import Result
from concert.quantities import q, Quantity
from concert.measures import rotation_axis, estimate_alignment_parameters
from concert.optimization import halver, optimize_parameter
from concert.imageprocessing import flat_correct, find_needle_tips, find_sphere_centers_corr
from concert.helpers import expects, is_iterable, Numeric
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.devices.shutters.base import Shutter
from concert.devices.cameras.base import Camera
from concert.ext.viewers import PyQtGraphViewer
from concert.progressbar import wrap_iterable
from concert.typing import ArrayLike, FramesProducer_T, CoroFunc_T


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


# TODO: A dummy camera does not know, how to behave when a dummy shutter is provided. It is easier
# to provide None for shutter. Enable before merge.
# @expects(Camera, RotationMotor, num_frames=Numeric(1), shutter=Shutter,
#          flat_motor=LinearMotor, flat_position=Numeric(1, q.m), y_0=Numeric(1),
#          y_1=Numeric(1))
async def acquire_frames(camera, rotation_motor, num_frames, shutter=None, flat_motor=None,
                         flat_position=None, y_0=0, y_1=None):
    """
    acquire_frames(camera, rotation_motor, num_frames, shutter=None, flat_motor=None,
                       flat_position=None, y_0=0, y_1=None)

    Acquire frames either for 360 degrees of rotation when num_frames is greater than 1 and a single
    frame otherwise.
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
            for _ in range(num_frames):
                # Only acquire for 360 degrees of rotation if num_frames specifies a number greater
                # than 1.
                if num_frames > 1:
                    await rotation_motor.move(2 * np.pi / num_frames * q.rad)
                await camera.trigger()
                frame = (await camera.grab())[y_0:y_1].astype(float)
                # TODO: When we are working dummy motor and there is no shutter available we
                # initialize flat and dark such that zero-division error is avoided during flat
                # field correction. Remove before merge.
                if shutter is None:
                    flat = np.ones(frame.shape)
                    dark = np.zeros(frame.shape)
                frame = flat_correct(frame, flat, dark=dark)
                frame = np.nan_to_num(-np.log(frame))
                # Huge numbers can also cause trouble
                frame[np.abs(frame) > 1e6] = 0
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
        frame_producer = acquire_frames(camera, rotation_motor, num_frames, shutter=shutter,
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
# Revised Alignment Routines
####################################################################################################
@dataclass
class AcquisitionDevices:
    """
    Encapsulates relevant devices which are collectively used for frame acquisition for alignment.

    - reference to camera
    - reference to shutter
    - reference to tomographic rotation motor
    - reference to linear motor moving tomographic rotation stage horizontally
    - reference to linear motor moving tomographic rotation stage vertically
    """
    camera: Camera
    shutter: Shutter
    tomo_motor: RotationMotor
    flat_motor: LinearMotor
    z_motor: LinearMotor


@dataclass
class AcquisitionContext:
    """
    Encapsulates devices and parameters for frame acquisition for alignment.

    - reference to devices which are relevant for acquiring frames using camera
    - height of the projections
    - width of the projections
    - flag indicating if flat field correction should be done for the acquired frames
    - image processing function to apply on each frame
    - flag indicating if absorptivity needs to ve calculated
    - optional position of the flat motor to move sample away from beam (only relevant if
    `flat_field_correct` is true)
    """
    devices: AcquisitionDevices
    height: int
    width: int
    flat_field_correct: bool
    absorptivity: bool
    num_frames: int = 1
    flat_position: Optional[Quantity] = None


@dataclass
class AlignmentDevices:
    """
    Encapsulates relevant devices for alignment fo which we might need to make frequent small
    adjustments.

    - rotation motor for pitch angle correction
    - rotation motor for roll angle correction
    - linear alignment motor to move sample horizontally parallel to the beam
    - linear alignment motor to move sample horizontally orthogonal to the beam
    """
    rot_motor_pitch: RotationMotor
    rot_motor_roll: RotationMotor
    align_motor_pbd: Optional[LinearMotor]
    align_motor_obd: Optional[LinearMotor]


@dataclass
class AlignmentContext:
    """
    Encapsulates devices and parameters for for alignment method.

    - reference to the devices, which are relevant for alignment of tomographic stage
    - pixel size in micrometer
    - sphere radius needed for tracking during ellipse-fit
    - max iterations for alignment
    - pixel sensitivity to derive a metric to evaluate alignment
    - initial gain to consider on first iteration
    - ceiling value to prevent explosive gain
    - epsilon value for difference in motor positions in consecutive iterations
    - epsilon value for difference in estimated angles in consecutive iterations
    - ceiling value for relative movement to prevent explosive motor movement
    - angular offset to be applied to tomographic rotation motor for correction
    - off-centering distance for alignment motor moving orthogonal to beam
    - off-centering distance for alignment motor moving parallel to beam
    - linear delta movement to determine correct direction
    - angular delta movement to determine correct direction
    - epsilon pixel error tolerance, needed especially during centering the sample
    - backlash compensation relative distance for linear motors
    - backlash compensation relative angle for rotation motors
    - image processing function to apply on the acquired frame
    - optional viewer to display processed frame
    - method to use for deriving offsets, one of ["phase_cross_corr", "template_match"]
    """
    devices: AlignmentDevices
    pixel_size_um: Quantity
    sphere_radius: int = 65
    max_iterations: int = 10
    pixel_sensitivity: int = 2
    init_gain: float = 1.0
    ceil_gain: float = 3.0
    eps_pose_diff: Quantity = 0.01 * q.deg
    eps_ang_diff: Quantity = 1e-7 * q.deg
    ceil_rel_move: Quantity = 1 * q.deg
    offset_rot_tomo: Quantity = 0 * q.deg
    offset_lin_obd: Quantity = 0.2 * q.mm
    offset_lin_pbd: Quantity = 0.2 * q.mm
    delta_move_mm: Quantity = 0.1 * q.mm
    delta_move_deg: Quantity = 0.1 * q.deg
    pixel_err_eps: float = 2.0
    bl_comp_rel_lin: Quantity = 0.1 * q.mm
    bl_comp_rel_rot: Quantity = 0.1 * q.deg
    proc_func: Callable[[ArrayLike], ArrayLike] = lambda x: x
    viewer: Optional[PyQtGraphViewer] = None
    offset_method: str = "phase_cross_corr"

    PHASE_CROSS_CORR: str = "phase_cross_corr"
    TEMPLATE_MATCH: str = "template_match"


def get_noop_logger() -> logging.Logger:
    """Provides a no-op logger"""
    logger = logging.getLogger("no-op")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


async def extract_ellipse_points(producer: FramesProducer_T, radius: int) -> List[ArrayLike]:
    """
    Finds sphere centers from incoming frames using correlation.
    
    :param producer: asynchronous producer of projections
    :type producer: `concert.typing.FramesProducer_T`
    :param radius: radius of the sphere in pixels
    :type radius: int
    :return: extracted ellipse centroids
    :rtype: List[`concert.typing.ArrayLike`]
    """
    return await find_sphere_centers_corr(producer, radius=radius)


async def set_best_position(
        motor: RotationMotor, rot_type: str, history: List[Dict[str, Quantity]],
        logger: logging.Logger) -> None:
    """
    Sets the motor position against the lowest angular error.

    :param motor: rotation motor for roll or pitch correction
    :type motor: `concert.devices.motors.base.RotationMotor`
    :param rot_type: rotation type, 'roll' or 'pitch'
    :type rot_type: str
    :param history: mapping of estimated angle error and respective motor movement from all
    iterations
    :type history: List[Dict[str, `concert.quantities.Quantity`]]
    :param logger: logger, defaults to no-op logger from caller
    :type logger: logging.Logger
    """
    positions, angles = list(zip(*[(item["position"], item[rot_type]) for item in history]))
    best_index = np.argmin(np.abs([angle.to_base_units().magnitude for angle in angles]))
    logger.debug(
        f"best: {rot_type} => position: {positions[best_index].to(q.deg)} " + 
        f"angle: {angles[best_index].to(q.deg)}")
    try:
        await motor.set_position(positions[best_index])
    except Exception as exc:
        logger.debug(f"{rot_type} motor could not set best position: {exc}")


async def make_step(
        iteration: int, motor: RotationMotor, pose_last: Quantity, ang_last: Quantity,
        ang_curr: Quantity, rot_type: str, align_ctx: AlignmentContext,
        logger: logging.Logger) -> Tuple[Quantity, Quantity]:
    """
    Makes a single iterative step with dynamic gain.

    :param iteration: current iteration
    :type iteration: int
    :param motor: rotation motor for roll or pitch correction
    :type motor: `concert.devices.motors.base.RotationMotor`
    :param pose_last: motor position from previous iteration
    :type pose_last: `concert.quantities.Quantity`
    :param ang_last: estimated angle in previous iteration
    :type ang_last: `concert.quantities.Quantity`
    :param ang_curr: estimated angle in current iteration
    :type ang_curr: `concert.quantities.Quantity`
    :param rot_type: rotation type, 'roll' or 'pitch'
    :type rot_type: str
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param logger: logger, defaults to no-op logger from caller
    :type logger: logging.Logger
    :return: estimated angle error and respective motor movement
    :rtype: Tuple[`concert.quantities.Quantity`, `concert.quantities.Quantity`]
    """
    pose_curr = await motor.get_position()
    logger.debug(
        f"iteration: {iteration} - {rot_type} motor-position: {pose_curr}")
    ang_diff = ang_curr - ang_last
    pose_diff = pose_curr - pose_last
    # If we are on the first iteration we use the initial gain to kick start the alignment, else
    # we try to compute the gain from position and angular differences.
    if iteration > 0:
        # If the angle change between previous and current iteration is significant then we compute
        # the dynamic gain as (position change / unit angle change) capped by `ceil_gain`
        # to avoid gigantic move.
        if abs(ang_diff) > align_ctx.eps_ang_diff:
            gain = np.clip(pose_diff / ang_diff, -align_ctx.ceil_gain, align_ctx.ceil_gain)
            logger.debug(f"gain(theoretical): ({pose_curr} - {pose_last}) " +
                         f"/ ({ang_curr} - {ang_last}) = " + f"{pose_diff / ang_diff}")
            logger.debug(f"gain(clipped) = {gain}")
        else:
            # If the angle has not changed significantly in subsequent iterations we don't make any
            # adjustments.
            gain = 0
            logger.debug(f"insignificant angular change, gain: {gain}")
    else:
        gain = align_ctx.init_gain
        logger.debug(f"first-iteration, initial gain used: {gain}")
    pose_last = pose_curr
    ang_last = ang_curr
    # Relative movement is capped by `ceil_rel_move`.
    move_rel: float = np.clip(
            -gain * ang_curr.magnitude,
            -align_ctx.ceil_rel_move.magnitude,
            align_ctx.ceil_rel_move.magnitude)
    logger.debug(f"{rot_type} motor-move(theoretical): (-{gain} * {ang_curr}) = {-gain * ang_curr}")
    logger.debug(f"{rot_type} motor-move(clipped): {move_rel}")
    if np.any(np.sign(move_rel)):
        try:
            await motor.move(move_rel * q.deg)
        except LimitError:
            logger.debug(f"{rot_type} motor encountered soft-limit error")
            return pose_last, ang_last
    else:
        logger.debug(f"{rot_type} motor didn't move")
        return pose_last, ang_last
    pose_curr = await motor.get_position()
    logger.debug(f"{rot_type} motor moved, current position: {pose_curr}")
    return pose_last, ang_last


@background
async def align_rotation_stage_ellipse_fit(
    acq_ctx: AcquisitionContext, align_ctx: AlignmentContext,
    logger: logging.Logger = get_noop_logger(), coro_func: Optional[CoroFunc_T] = None) -> None:
    """
    Implements dynamically gained iterative alignment routine for pitch angle and roll angle
    correction based on ellipse-fit. This implementation incorporates the idea of approaching the
    alignment by adjusting the rotation stage in small steps. In each iteration we make a step for
    roll angle correction followed by a step for pitch angle correction. Estimated angular
    correction is translated to a motor movement which is safe-guarded by a ceiling to prevent an
    explosive movement that might push the system to an anomaly.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param logger: optional logger
    :type logger: logging.Logger
    :param coro_func: optional awaitable callback function
    :type coro_func: Optional[`concert.typing.CoroFunc_T`]
    """
    func_name = "align_rotation_stage_ellipse_fit" 
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    logger.debug(f"Start: {func_name}")
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    # Initialize alignment-related variables and metric.
    pitch_ang_last: Quantity = 0 * q.deg
    pitch_pose_last: Quantity = await align_ctx.devices.rot_motor_pitch.get_position()
    pitch_can_continue = True
    pitch_align_history: List[Dict[str, Quantity]] = []
    roll_ang_last: Quantity = 0 * q.deg
    roll_pose_last: Quantity = await align_ctx.devices.rot_motor_roll.get_position()
    roll_can_continue = True
    roll_align_history: List[Dict[str, Quantity]] = []
    frames_result = Result()
    for iteration in range(align_ctx.max_iterations):
        logger.debug("=" * 3 * len(f"Start: {func_name}"))
        acq_consumers = [functools.partial(extract_ellipse_points, radius=align_ctx.sphere_radius),
                         frames_result]
        iter_start = time.perf_counter()
        # Acquire frames from camera and perform flat-field correction.
        producer: FramesProducer_T = acquire_frames(
            acq_ctx.devices.camera, acq_ctx.devices.tomo_motor, acq_ctx.num_frames,
            shutter=acq_ctx.devices.shutter, flat_motor=acq_ctx.devices.flat_motor,
            flat_position=acq_ctx.flat_position)
        coros = broadcast(producer, *acq_consumers)
        # Setting tomographic rotation motor to zero degree after a full circular rotation is a
        # safety measure against potential malfunction.
        await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_rot_tomo)
        centroids = []
        try:
            centroids = (await asyncio.gather(*coros))[1]
        except Exception as tips_exc:
            logger.debug(
                f"{func_name}:iteration:{iteration}-error finding reference points: {tips_exc}")
            logger.debug(f"End: {func_name}")
            logger.debug("#" * len(f"End: {func_name}"))
            return
        logger.debug(
            f"found {len(centroids)} centroids, elapsed time: {time.perf_counter() - iter_start}")
        roll_ang_curr, pitch_ang_curr, _ = estimate_alignment_parameters(centroids=centroids)
        logger.debug(f"current pitch angle error: {pitch_ang_curr.magnitude}")
        logger.debug(f"current roll angle error: {roll_ang_curr.magnitude}")
        # Start alignment iteration
        # NOTE: Alignment metric is a measure of resolution sensitivity. We derive it as an angular
        # error threshold which represents maximum `align_ctx.pixel_sensitivity` pixels of
        # resolution loss vertically against `acq_ctx.width` pixels horizontally.
        align_metric: Quantity = np.rad2deg(
            np.arctan(align_ctx.pixel_sensitivity / acq_ctx.width)) * q.deg
        if pitch_can_continue:
            pitch_pose = await align_ctx.devices.rot_motor_pitch.get_position()
            pitch_align_history.append({"position": pitch_pose, "pitch": pitch_ang_curr})
            if abs(pitch_ang_curr) >= align_metric and (
                abs(pitch_pose_last - pitch_pose) >= align_ctx.eps_pose_diff or iteration == 0):
                pitch_pose_last, pitch_ang_last = await make_step(
                    iteration=iteration, motor=align_ctx.devices.rot_motor_pitch,
                    pose_last=pitch_pose_last, ang_last=pitch_ang_last, ang_curr=pitch_ang_curr,
                    rot_type="pitch", align_ctx=align_ctx, logger=logger)
            else:
                logger.debug(f"{func_name}: desired pitch angle threshold reached")
                pitch_can_continue = False
        if roll_can_continue:
            roll_pose = await align_ctx.devices.rot_motor_roll.get_position()
            roll_align_history.append({"position": roll_pose, "roll": roll_ang_curr})
            if abs(roll_ang_curr) >= align_metric and (
                abs(roll_pose_last - roll_pose) >= align_ctx.eps_pose_diff or iteration == 0):
                roll_pose_last, roll_ang_last = await make_step(
                    iteration=iteration, motor=align_ctx.devices.rot_motor_roll,
                    pose_last=roll_pose_last, ang_last=roll_ang_last, ang_curr=roll_ang_curr,
                    rot_type="roll", align_ctx=align_ctx, logger=logger)
            else:
                logger.debug(f"{func_name}: desired roll angle threshold reached")
                roll_can_continue = False
        if coro_func:
            await coro_func()
        if not pitch_can_continue and not roll_can_continue:
            break
    if pitch_can_continue:
        logger.debug(f"{func_name}: max iterations reached but pitch error is still significant")
    if roll_can_continue:
        logger.debug(f"{func_name}: max iterations reached but roll error is still significant")
    # Move to the best known position
    coros = []
    coros.append(set_best_position(
        motor=align_ctx.devices.rot_motor_pitch, rot_type="pitch", history=pitch_align_history,
        logger=logger))
    coros.append(set_best_position(
        motor=align_ctx.devices.rot_motor_roll, rot_type="roll", history=roll_align_history,
        logger=logger))
    await asyncio.gather(*coros)
    # Leave the system in stable state
    logger.debug(
        f"final pitch = {await align_ctx.devices.rot_motor_pitch.get_position()} " +
        f"final roll = {await align_ctx.devices.rot_motor_roll.get_position()}")
    logger.debug(f"End: {func_name}")
####################################################################################################


class ProcessError(Exception):

    """
    Exception raised by a process when something goes wrong with the procedure
    it tries to accomplish, e.g. cannot focus, cannot align rotation axis, etc.

    """
    pass



