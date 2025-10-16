import asyncio
from dataclasses import dataclass
import functools
import time
from itertools import product
from functools import reduce
import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
import scipy.ndimage as sdi
import skimage.filters as sfl
import skimage.feature as sft
import skimage.measure as sms
from skimage.measure._regionprops import RegionProperties
import skimage.registration as skr
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
from concert.progressbar import wrap_iterable
from concert.typing import ArrayLike, FrameProducer_T, FramesProducer_T, CoroFunc_T, Motor_T


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
    Encapsulates relevant devices which are frequently needed for acquisition

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
class AcquisitionParams:
    """
    Encapsulates parameters relevant for projections acquisition

    - number of frames to extract the sphere centroids from
    - position for flat-motor to move sample away from beam
    - radius of the alignment sphere marker for centroid estimation
    - height of the projections
    - width of the projections
    - sigma value for low-pass filtering, we deliberately choose a very high value to suppress
    the background as much as possible
    """
    num_frames: int = 10
    flat_position: Quantity = 7 * q.mm
    sphere_radius: int = 65
    height: int = 2016
    width: int = 2016
    sigma: int = 20
    
    @property
    def align_metric(self) -> Quantity:
        """
        Provides a metric to evaluate alignment.
        
        Represents 1 pixel of resolution sensitivity. It means the angular threshold which
        represents maximum 1 pixel of resolution loss vertically against projection width pixels
        horizontally.

        :return: angular threshold for 1 pixel of resolution sensitivity
        :rtype: `concert.quantities.Quantity`
        """
        return np.rad2deg(np.arctan(1 / self.width)) * q.deg


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
class AlignmentParams:
    """
    Encapsulates parameters required for alignment

    - max iterations for alignment
    - initial gain to consider on first iteration
    - ceiling value to prevent explosive gain
    - epsilon value for difference in motor positions in consecutive iterations
    - epsilon value for difference in estimated angles in consecutive iterations
    - ceiling value to prevent explosive motor movement
    - off-centering distance for alignment motor moving orthogonal to beam
    - off-centering distance for alignment motor moving parallel to beam
    - linear delta movement to determine correct direction
    - angular delta movement to determine correct direction
    - epsilon pixel error tolerance, needed especially during centering the sample
    - backlash compensation relative distance for linear motors
    - backlash compensation relative angle for rotation motors
    - backlash compensation absolute distance for linear motors
    - backlash compensation absolute angle for rotation motors
    - use binary image to compute sample offsets
    """
    max_iterations: int = 50
    init_gain: float = 1.0
    ceil_gain: float = 5.0
    eps_pose: Quantity = 0.01 * q.deg
    eps_ang_diff: Quantity = 1e-7 * q.deg
    ceil_rel_move: Quantity = 1 * q.deg
    offset_obd: Quantity = 0.2 * q.mm
    offset_pbd: Quantity = 0.4 * q.mm
    delta_move_mm: Quantity = 0.01 * q.mm
    delta_move_deg: Quantity = 0.1 * q.deg
    px_eps: float = 2.0
    bl_comp_rel_lin: Quantity = 0.1 * q.mm
    bl_comp_rel_rot: Quantity = 0.1 * q.deg
    use_binary: bool = True


@dataclass
class AlignmentState:
    """
    Encapsulates elements of the state management for the alignment

    - checkpoints dictionary tracks last known positions of the motors involved in the alignment.
    - patches dictionary contains a patch of our sample for each of the terminal angles.
    - baseline_scores contains the estimated certainty scores for the sample definitely being
    inside FOV.
    - uncertainty_threshold is the maximum uncertainty to allow to conclude that the sample is in
    fact inside FOV.

    TODO: Checkpoint based system is not fully implemented yet. It is supposed to serve state
    management during alignment and help in recovering from anomalies like sample going outside FOV.
    The idea for the checkpoints dictionary is to track the last known 'good' motor positions for
    which sample was definitely in FOV. In conjunction to that we need a stack which tracks the
    adjustments / relative movements made to the respective motors in order. At any given point in
    time if we detect the sample outside FOV we can pop the last movement from the stack which has
    caused it and try to recover from that.
    """
    checkpoints: Dict[str, Quantity]
    patches: Dict[str, ArrayLike]
    baseline_scores: Dict[str, float]
    dim: int = 200
    uncertainty_threshold: float = 0.2

    def __str__(self) -> str:
        val = "Motor Positions:\n"
        for key, value in self.checkpoints.items():
            val += f" {key} = {value}\n"
        val += "Baseline Scores (expected close to 1.0):\n"
        for key, value in self.baseline_scores.items():
            val += f" {key} degree = {value}\n"
        return val

    def sample_in_FOV(self, frame: ArrayLike, angle: int) -> bool:
        """
        Evaluates if sample is inside FOV for the given `frame` and `angle`.

        We derive a cofidence score by matching a template patch for the given `angle` to the
        `frame`. This score is then used to compute a relative certainty against respective baseline
        score for the `angle` and from that we compute an uncertainty metric, which is compared
        against the configured threshold. In principle, we are asking, how uncertain we are that the
        sample is inside FOV against the maximum uncertainty that we want to allow.

        :param frame: projection to evaluate
        :type frame: `concert.typing.ArrayLike`
        :param angle: angle to select the patch and score
        :type angle: int
        :return: if smaple inside FOV
        :rtype: bool
        """
        score = np.max(sft.match_template(
            image=frame, template=self.patches[str(angle)], pad_input=True))
        relative_likelihood = (score / self.baseline_scores[str(angle)])
        uncertainty = 1 - relative_likelihood
        return uncertainty < self.uncertainty_threshold


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
    :rtype: List[ArrayLike]
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
        ang_curr: Quantity, rot_type: str, align_params: AlignmentParams,
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
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
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
        if abs(ang_diff) > align_params.eps_ang_diff:
            gain = np.clip(pose_diff / ang_diff, -align_params.ceil_gain, align_params.ceil_gain)
            logger.debug(f"gain(theoretical): ({pose_curr} - {pose_last}) " +
                         f"/ ({ang_curr} - {ang_last}) = " + f"{pose_diff / ang_diff}")
            logger.debug(f"gain(clipped) = {gain}")
        else:
            # If the angle has not changed significantly in subsequent iterations we don't make any
            # adjustments.
            gain = 0
            logger.debug(f"insignificant angular change, gain: {gain}")
    else:
        gain = align_params.init_gain
        logger.debug(f"first-iteration, initial gain used: {gain}")
    pose_last = pose_curr
    ang_last = ang_curr
    # Relative movement is capped by `ceil_rel_move`.
    move_rel: float = np.clip(
            -gain * ang_curr.magnitude,
            -align_params.ceil_rel_move.magnitude,
            align_params.ceil_rel_move.magnitude)
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
    acq_devices: AcquisitionDevices, acq_params: AcquisitionParams,
    align_devices: AlignmentDevices, align_params: AlignmentParams,
    logger: logging.Logger = get_noop_logger(), coro_func: Optional[CoroFunc_T] = None) -> None:
    """
    Implements dynamically gained iterative alignment routine for pitch angle and roll angle
    correction based on ellipse-fit. This implementation incorporates the idea of approaching the
    alignment by adjusting the rotation stage in small steps. In each iteration we make a step for
    roll angle correction followed by a step for pitch angle correction. Estimated angular
    correction is translated to a motor movement which is safe-guarded by a ceiling to prevent an
    explosive movement that might push the system to an anomaly.

    :param acq_devices: devices, which are collectively required for acquisition
    :type acq_devices: `concert.processes.common.AcquisitionDevices`
    :param acq_params: configurations specific to acquisition
    :type acq_params: `concert.processes.common.AcquisitionParams`
    :param align_devices: devices which are collectively required for alignment
    :type align_devices: `concert.processes.common.AlignmentDevices`
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
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
    pitch_pose_last: Quantity = await align_devices.rot_motor_pitch.get_position()
    pitch_can_continue = True
    pitch_align_history: List[Dict[str, Quantity]] = []
    roll_ang_last: Quantity = 0 * q.deg
    roll_pose_last: Quantity = await align_devices.rot_motor_roll.get_position()
    roll_can_continue = True
    roll_align_history: List[Dict[str, Quantity]] = []
    frames_result = Result()
    for iteration in range(align_params.max_iterations):
        logger.debug("=" * 3 * len(f"Start: {func_name}"))
        acq_consumers = [functools.partial(extract_ellipse_points, radius=acq_params.sphere_radius),
                         frames_result]
        iter_start = time.perf_counter()
        # Acquire frames from camera and perform flat-field correction.
        producer: FramesProducer_T = acquire_frames(
            acq_devices.camera, acq_devices.tomo_motor, acq_params.num_frames,
            shutter=acq_devices.shutter, flat_motor=acq_devices.flat_motor,
            flat_position=acq_params.flat_position)
        coros = broadcast(producer, *acq_consumers)
        # Setting tomographic rotation motor to zero degree after a full circular rotation is a
        # safety measure against potential malfunction.
        await acq_devices.tomo_motor.set_position(0 * q.deg)
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
        if pitch_can_continue:
            pitch_pose = await align_devices.rot_motor_pitch.get_position()
            pitch_align_history.append({"position": pitch_pose, "pitch": pitch_ang_curr})
            if abs(pitch_ang_curr) >= acq_params.align_metric and (
                abs(pitch_pose_last - pitch_pose) >= align_params.eps_pose or iteration == 0):
                pitch_pose_last, pitch_ang_last = await make_step(
                    iteration=iteration, motor=align_devices.rot_motor_pitch,
                    pose_last=pitch_pose_last, ang_last=pitch_ang_last, ang_curr=pitch_ang_curr,
                    rot_type="pitch", align_params=align_params, logger=logger)
            else:
                logger.debug(f"{func_name}: desired pitch angle threshold reached")
                pitch_can_continue = False
        if roll_can_continue:
            roll_pose = await align_devices.rot_motor_roll.get_position()
            roll_align_history.append({"position": roll_pose, "roll": roll_ang_curr})
            if abs(roll_ang_curr) >= acq_params.align_metric and (
                abs(roll_pose_last - roll_pose) >= align_params.eps_pose or iteration == 0):
                roll_pose_last, roll_ang_last = await make_step(
                    iteration=iteration, motor=align_devices.rot_motor_roll,
                    pose_last=roll_pose_last, ang_last=roll_ang_last, ang_curr=roll_ang_curr,
                    rot_type="roll", align_params=align_params, logger=logger)
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
        motor=align_devices.rot_motor_pitch, rot_type="pitch", history=pitch_align_history,
        logger=logger))
    coros.append(set_best_position(
        motor=align_devices.rot_motor_roll, rot_type="roll", history=roll_align_history,
        logger=logger))
    await asyncio.gather(*coros)
    # Leave the system in stable state
    logger.debug(
        f"final pitch = {await align_devices.rot_motor_pitch.get_position()} " +
        f"final roll = {await align_devices.rot_motor_roll.get_position()}")
    logger.debug(f"End: {func_name}")


def make_binary(
        projection: ArrayLike, sigma: int, logger: logging.Logger = get_noop_logger()) -> ArrayLike:
    """
    Exploits the high absorption of the sample to make a binary image from the projection
    
    NOTE: We use try-except block here to be able to work with simulation camera. For real
    acquisitions an exception inside this block is unlikely.

    :param projection: projection
    :type projection: `concert.typing.ArrayLike`
    :param sigma: sigma value for low-pass filtering
    :type sigma: int
    """
    logger.debug = print # TODO 
    try:
        img: ArrayLike = projection.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        smoothed: ArrayLike = sfl.gaussian(img, sigma=sigma)
        seg: float = sfl.threshold_otsu(smoothed)
        mask: ArrayLike = smoothed > seg
        if projection.shape == mask.shape:
            return mask
    except ValueError:
        logger.debug("make_binary: could not create mask, falling back to intensity thresholding")
    fallback: ArrayLike = projection.copy()
    fallback[projection < projection.max() / 2] = projection.min()
    return fallback


async def get_sample_shifts(
        acq_devices: AcquisitionDevices, acq_params: AcquisitionParams,
        align_params: AlignmentParams,  align_state: AlignmentState,
        start_ang_deg: Quantity) -> Tuple[float, float]:
    """
    Estimates the linear shifts of the sample across a 180 degrees rotation along given cartesian
    axis for which the function is called.

    A specific angular range is associated with each direction e.g., [0, 180] for horizontally 
    orthogonal axis to beam direction and [90, 270] for horizontally-parallel axis to beam
    direction. This mapping is not universal but the angular offset remains 180 degrees in either
    direction. We therefore cross-correlate the projections for both ends of 180 degrees rotation to
    measure the shifts.

    NOTE: We see the effect of the said rotation as linear shifts onto the projection plane, which
    itself is orthogonal to the XY plane in which rotation takes place. If we take the top-view
    perspective of rotation plane together with the projection plane, few geometric aspects would
    be apparent, which this function relies upon.

    - Alignment motor which moves linearly along the orthogonal axis to beam direction, takes the
    sample away from geometric center of rotation. Other alignment motor does the same along the
    parallel axis to beam direction. In a nutshell, the offset these movements make from center
    becomes the radius of the rotation w.r.t. that specific axis of this plane. Therefore, when we
    measure the horizontal shift of the sample from two projections taken 180 degrees apart we are
    measuring the diameter for the rotation. Halving this would give us the radius and if we move
    with alignment motors towards the center by this amount sample would be shifted close to center
    of rotation.

    - Alignment motors are placed on top of rotation motor, therefore they rotate. This is the
    reason, why we refrained from strictly associating cartesian-X or cartesian-Y to either of the
    axes of the rotation plane and tried to generalize them w.r.t to beam direction. A specific
    BeamLine may choose to adopt some convention for its CT geometry e.g., associating the [0, 180]
    rotation with the orthogonal axis to beam direction and [90, 270] with the parallel axis. In
    another scenario we might see the reverse association. Taking this specific association as our
    reference, if we move our sample along parallel axis to beam direction, while rotation motor is
    at 0 degree we won't perceive the displacement, because in this situation it has happened in
    the same direction as beam. But if we now rotate by 90 degrees and make the same movement,
    resulting displacement would be perceived because the axis along which we moved is now rotated
    and spanned orthogonally to the beam direction. This semantic is utilized during manual
    alignment.

    - Vertical shift measured by normalized cross correlation contributes one of the key components
    to estimate the pitch and roll angle errors, because they are directly proportional to vertical
    shifts in respective directions. We consider signed value for this to estimate directional
    corrective movement.

    :param acq_devices: devices, which are collectively required for acquisition
    :type acq_devices: `concert.processes.common.AcquisitionDevices`
    :param acq_params: configurations specific to acquisition
    :type acq_params: `concert.processes.common.AcquisitionParams`
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
    :param align_state: alignment state
    :type align_state: `concert.processes.common.AlignmentState`
    :param start_angle: initial angle(degrees) to set before measuring offset
    :type start_angle: `concert.quantities.Quantity`
    :return: vertical shift of sample caused by misalignment and distance of sample from center
    of rotation
    :rtype: Tuple[float, float]
    """
    await acq_devices.tomo_motor.set_position(start_ang_deg)
    producer: FrameProducer_T = acquire_frames(
        acq_devices.camera, acq_devices.tomo_motor, 1, acq_devices.shutter, acq_devices.flat_motor,
        acq_params.flat_position)
    ref_frame: ArrayLike = np.asarray([frame async for frame in producer][0])
    if not align_state.sample_in_FOV(frame=ref_frame, angle=start_ang_deg.magnitude):
        raise ProcessError("sample went outside FOV before rotation, aborting")
    await acq_devices.tomo_motor.move(180 * q.deg)
    producer =  acquire_frames(
        acq_devices.camera, acq_devices.tomo_motor, 1, acq_devices.shutter, acq_devices.flat_motor,
        acq_params.flat_position)
    mov_frame: ArrayLike = np.asarray([frame async for frame in producer][0])
    if not align_state.sample_in_FOV(frame=mov_frame, angle=start_ang_deg.magnitude + 180):
        raise ProcessError("sample went outside FOV after rotation, aborting")
    ref_img: ArrayLike = make_binary(projection=ref_frame, sigma=acq_params.sigma) \
        if align_params.use_binary else ref_frame
    mov_img: ArrayLike = make_binary(projection=mov_frame, sigma=acq_params.sigma) \
        if align_params.use_binary else mov_frame
    shift_yx, _, _ = skr.phase_cross_correlation(
        reference_image=ref_img, moving_image=mov_img, upsample_factor=4)
    await acq_devices.tomo_motor.move(-180 * q.deg)
    return shift_yx[0], abs(shift_yx[1]) / 2


async def init_alignment_state(
        acq_devices: AcquisitionDevices, acq_params: AcquisitionParams,
        align_devices: AlignmentDevices) -> AlignmentState:
    """
    Initializes alignment state

    - STEP 1: Record relevant motor positions before starting with the alignment
    - STEP 2: For each terminal angle, grab a frame, extract a patch containing the sample, use
    template matching to get a baseline certainty score for sample definitely being inside FOV.

    :param acq_devices: devices, which are collectively required for acquisition
    :type acq_devices: `concert.processes.common.AcquisitionDevices`
    :param acq_params: configurations specific to acquisition
    :type acq_params: `concert.processes.common.AcquisitionParams`
    :param align_devices: devices which are collectively required for alignment
    :type align_devices: `concert.processes.common.AlignmentDevices`
    :return: initial alignment state
    :rtype: `concert.processes.common.AlignmentState`
    """
    state = AlignmentState(checkpoints={}, patches={}, baseline_scores={})
    # Record all relevant motor positions
    for motor_str in ["flat_motor", "z_motor"]:
        state.checkpoints[motor_str] = await getattr(acq_devices, motor_str).get_position()
    for motor_str in ["align_motor_obd", "align_motor_pbd", "rot_motor_roll", "rot_motor_pitch"]:
        state.checkpoints[motor_str] = await getattr(align_devices, motor_str).get_position()
    # Record patches and baseline scores for the terminal angles
    for angle in [0, 90, 180, 270]:
        await acq_devices.tomo_motor.set_position(angle * q.deg)
        producer: FrameProducer_T = acquire_frames(
            acq_devices.camera, acq_devices.tomo_motor, 1, acq_devices.shutter,
            acq_devices.flat_motor, acq_params.flat_position)
        frame: ArrayLike = np.asarray([frame async for frame in producer][0])
        mask: ArrayLike = make_binary(projection=frame, sigma=acq_params.sigma)
        regions: List[RegionProperties] = sms.regionprops(label_image=sms.label(mask))
        cnt_y, cnt_x = sorted(regions, key=lambda r: r.area, reverse=True)[0].centroid
        dim = state.dim // 2
        state.patches[str(angle)] = frame[
            int(cnt_y) - dim:int(cnt_y) + dim, int(cnt_x)- dim:int(cnt_x) + dim]
        state.baseline_scores[str(angle)] = np.max(
            sft.match_template(image=frame, template=state.patches[str(angle)], pad_input=True))
    await acq_devices.tomo_motor.set_position(0 * q.deg)
    return state


async def move_relative(
        motor: Motor_T, move_by: Quantity,
        align_params: AlignmentParams, logger: logging.Logger = get_noop_logger()) -> None:
    """
    Makes a backlash-compensated relative move.

    Backlash refers to the small amount of lost motion or mechanical slack that occurs whenever a
    motion system reverses direction. In high-precision linear or rotary stages, it arises from
    clearances between mating components such as screw threads, gears, or couplings. When direction
    changes, the driving element must take up this slack before the driven part begins to move,
    resulting in a temporary position error. It can prevent us from converging to minima of the
    angular errors and should be taken into account for motor movement.

    To compensate for backlash we take a uni-directional approach with an assumption, that any +ve
    directional movement of the motor is feasible without the mechanical slack. When we change
    direction from +ve to -ve or vice versa we need to take backlash into account. Our objective is
    to make relative movements in a way that backlash remains constant and consistently in the -ve
    direction when we make the final move which always has to be in the +ve direction according to
    our assumption.

    To account for backlash we overshoot towards -ve direction by a small distance `bl_comp_rel_rot`
    or `bl_comp_rel_lin` in relative sense and then come back by the same amount toward the +ve
    direction. These small relative distances should be big-enough to cover the systematic backlash.
    Accuracy in motor movement in this manner relies on the preload step at the begining, where we
    try to ensure that gears are touching the face in the +ve direction while backlash remains in
    the -ve direction.

    :param motor: linear or rotation motor used for alignment
    :type motor: `concert.typing.Motor_T`
    :param move_by: relative distance to move the motor
    :type move_by: `concert.quantities.Quantity`
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later
    if np.sign(move_by) > 0:
        await motor.move(move_by)
        return
    bl_comp: Quantity = align_params.bl_comp_rel_rot if isinstance(motor, RotationMotor) \
        else align_params.bl_comp_rel_lin
    await motor.move(move_by - bl_comp)
    await motor.move(bl_comp)


async def center_sample_on_axis(
        acq_devices: AcquisitionDevices, acq_params: AcquisitionParams,
        align_devices: AlignmentDevices, align_params: AlignmentParams,
        align_state: AlignmentState, pixel_size_um: Quantity,
        logger: logging.Logger = get_noop_logger()) -> None:
    """
    Adjusts alignment motors orthogonal to the beam direction, `align_motor_obd` and parallel to the
    beam direction, `align_motor_pbd` to center the sample on rotation axis. This is prerequisite
    to aligning the rotation stage.

    Alignment motors are linear motors which seat on top of tomographic rotation motor on stage. The
    stage itself can be moved into or out of the beam using `flat_motor` (horizontally orthogonal
    to beam direction). Rotation axis is the geometric center of the rotation motor. Since alignment
    motors, `align_motor_pbd` and `align_motor_obd` seat on top the rotation motor they directly
    influence the rotation radius i.e., how far from the center a sample is rotating. We project a
    rotation in 3D cartesian system onto 2D plane to measure and reduce rotation radius in iterative
    steps.

    In 3D, sample rotates w.r.t vertical Z-axis (0, 0, 1), while `align_motor_obd` displaces the
    sample along X-axis (1, 0, 0) and `align_motor_obd` displaces the sample along Y-axis (0, 1, 0).
    In this context X-axis is spanned horizontally-orthogonal to the beam direction and Y-axis is
    spanned horizontally-parallel to the beam direction. Although this mapping is not universal,
    logic to estimate the rotation radius remains similar, because a specific angular offset is
    associated with each direction and respective alignment motor e.g., [0, 180] is for
    `align_motor_obd` and [90, 270] is for `align_motor_pbd`.
    
    From high level POV we want to measure how far the sample is from axis of rotation and adjust
    the motor to move the sample toward the axis from both horizontal directions using respective
    motors. The rotation radius computed from both directions may not be same because displacement
    of these motors are independent of each other but at the end of centering the sample on
    rotation axis, we should not perceive any displacement from rotation.
    
    On first iteration we determine the sign of correct motor movement by making a small move
    `delta_move_mm` in either direction and inspecting if that reduces the rotation radius in that
    direction. If it does then we have the correct sign, otherwise we toggle the sign.

    :param acq_devices: devices, which are collectively required for acquisition
    :type acq_devices: `concert.processes.common.AcquisitionDevices`
    :param acq_params: configurations specific to acquisition
    :type acq_params: `concert.processes.common.AcquisitionParams`
    :param align_devices: devices which are collectively required for alignment
    :type align_devices: `concert.processes.common.AlignmentDevices`
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
    :param align_state: alignment state
    :type align_state: `concert.processes.common.AlignmentState`
    :param pixel_size_um: pixel size in microns
    :type pixel_size_um: `concert.quantities.Quantity`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later

    async def _move_corrective(motor: LinearMotor, offset: float, start_ang_deg: Quantity) -> float:
        # await motor.move(align_params.delta_move_mm)
        await move_relative(
            motor=motor, move_by=align_params.delta_move_mm,
            align_params=align_params, logger=logger)
        _, _offset = await get_sample_shifts(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
            align_state=align_state, start_ang_deg=start_ang_deg)
        # await motor.move(-align_params.delta_move_mm)
        await move_relative(
            motor=motor, move_by=-align_params.delta_move_mm,
            align_params=align_params, logger=logger)
        if _offset > offset:
            offset = -offset
        # await motor.move(offset * pixel_size_um)
        await move_relative(
            motor=motor, move_by=offset * pixel_size_um,
            align_params=align_params, logger=logger)
        _, _new_offset = await get_sample_shifts(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
            align_state=align_state, start_ang_deg=start_ang_deg)
        return _new_offset
    
    # Get initial distances from center of rotation
    _, offset_obd = await get_sample_shifts(
        acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
        align_state=align_state, start_ang_deg=0 * q.deg)
    _, offset_pbd = await get_sample_shifts(
        acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
        align_state=align_state, start_ang_deg=90 * q.deg)
    logger.debug(f">> before: offset_obd = {offset_obd} offset_pbd = {offset_pbd}")
    while offset_obd > align_params.px_eps or offset_pbd > align_params.px_eps:
        # Make step adjustment for alignment motor orthogonal to beam direction
        if offset_obd > align_params.px_eps:
            offset_obd = await _move_corrective(
                motor=align_devices.align_motor_obd, offset=offset_obd, start_ang_deg=0 * q.deg)
        # Make step adjustment for alignment motor parallel to beam direction.
        if offset_pbd > align_params.px_eps:
            offset_pbd = await _move_corrective(
                motor=align_devices.align_motor_pbd, offset=offset_pbd, start_ang_deg=90 * q.deg)
    logger.debug(f">> after: offset_obd = {offset_obd} offset_pbd = {offset_pbd}")


async def offset_from_projection_center(
        acq_devices: AcquisitionDevices, acq_params: AcquisitionParams,
        align_params: AlignmentParams) -> Tuple[float, float]:
    """
    Derives the distance in pixels between geometric center of the projection and center of mass
    in XY detector plane. These values are useful to adjust tomographic stage motor (horizontally
    orthogonal to beam direction) and z-motor (vertically orthogonal to beam direction) to center a
    sample in the projection.

    :param acq_devices: devices, which are collectively required for acquisition
    :type acq_devices: `concert.processes.common.AcquisitionDevices`
    :param acq_params: configurations specific to acquisition
    :type acq_params: `concert.processes.common.AcquisitionParams`
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
    :return: distance between the geometric center of projection and center of mass 
    :rtype: Tuple[float, float]
    """
    producer: FrameProducer_T = acquire_frames(
        acq_devices.camera, acq_devices.tomo_motor, 1, acq_devices.shutter, acq_devices.flat_motor,
        acq_params.flat_position)
    frame: ArrayLike = np.asarray([frame async for frame in producer][0])
    ref_img: ArrayLike = make_binary(frame, sigma=acq_params.sigma) \
        if align_params.use_binary else frame
    ycm, xcm = sdi.center_of_mass(ref_img)
    yc_proj, xc_proj = frame.shape[0] / 2, frame.shape[1] / 2
    z_offset, stage_offset = abs(yc_proj - ycm), abs(xc_proj - xcm)
    return z_offset, stage_offset


async def center_stage_in_projection(
        acq_devices: AcquisitionDevices, acq_params: AcquisitionParams,
        align_params: AlignmentParams, pixel_size_um: Quantity,
        logger: logging.Logger = get_noop_logger()) -> None:
    """
    Adjusts the vertical motor(`z_motor`) and horizontal motor(`flat_motor`) to put the center of
    mass of the sample in the middle of the projection.
   
    :param acq_devices: devices, which are collectively required for acquisition
    :type acq_devices: `concert.processes.common.AcquisitionDevices`
    :param acq_params: configurations specific to acquisition
    :type acq_params: `concert.processes.common.AcquisitionParams`
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
    :param pixel_size_um: pixel size in microns
    :type pixel_size_um: `concert.quantities.Quantity`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later
    z_offset, stage_offset = await offset_from_projection_center(
        acq_devices=acq_devices, acq_params=acq_params, align_params=align_params)
    logger.debug(f">> before: z_offset = {z_offset} stage_offset = {stage_offset}")
    while z_offset > align_params.px_eps or stage_offset > align_params.px_eps:
        # Make step adjustment for z-motor (vertically orthogonal to beam direction)
        if z_offset > align_params.px_eps:
            # await acq_devices.z_motor.move(align_params.delta_move_mm)
            await move_relative(
                motor=acq_devices.z_motor, move_by=align_params.delta_move_mm,
                align_params=align_params, logger=logger)
            _z_offset, _ = await offset_from_projection_center(
                acq_devices=acq_devices, acq_params=acq_params, align_params=align_params)
            # await acq_devices.z_motor.move(-align_params.delta_move_mm)
            await move_relative(
                motor=acq_devices.z_motor, move_by=-align_params.delta_move_mm,
                align_params=align_params, logger=logger)
            if _z_offset > z_offset:
                z_offset = -z_offset
            # await acq_devices.z_motor.move(z_offset * pixel_size_um)
            await move_relative(
                motor=acq_devices.z_motor, move_by=z_offset * pixel_size_um,
                align_params=align_params, logger=logger)
        # Make step adjustment for stage motor (horizontally orthogonal to beam direction)
        if stage_offset > align_params.px_eps:
            # await acq_devices.flat_motor.move(align_params.delta_move_mm)
            await move_relative(
                motor=acq_devices.flat_motor, move_by=align_params.delta_move_mm,
                align_params=align_params, logger=logger)
            _, _stage_offset = await offset_from_projection_center(
                acq_devices=acq_devices, acq_params=acq_params, align_params=align_params)
            # await acq_devices.flat_motor.move(-align_params.delta_move_mm)
            await move_relative(
                motor=acq_devices.flat_motor, move_by=-align_params.delta_move_mm,
                align_params=align_params, logger=logger)
            if _stage_offset > stage_offset:
                stage_offset = -stage_offset
            # await acq_devices.flat_motor.move(stage_offset * pixel_size_um)
            await move_relative(
                motor=acq_devices.flat_motor, move_by=stage_offset * pixel_size_um,
                align_params=align_params, logger=logger)
        z_offset, stage_offset = await offset_from_projection_center(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params)
    logger.debug(f">> after: z_offset = {z_offset} stage_offset = {stage_offset}")


@background
async def align_rotation_stage_comparative(
    acq_devices: AcquisitionDevices, acq_params: AcquisitionParams,
    align_devices: AlignmentDevices, align_params: AlignmentParams,
    pixel_size_um: Quantity, logger: logging.Logger = get_noop_logger()) -> None:
    """
    Aligns rotation stage comparing the vertical positions of alignment phantom across half-circle
    rotations.

    NOTE: In this implementation we work with following conventions of our parallel-beam CT geometry
    and the relevant subset of the motors associated with the alignment.

    - `z_motor`: moves the tomo stage along vertically-orthogonal axis to beam direction.
    - `flat_motor`: moves the tomo stage along horizontally-orthogonal axis to beam direction.
    - `tomo_motor`: seats on top of tomo stage and performs 360 degrees of rotation. Axis of
        rotation is the geometric center of this motor.
    - `align_motor_obd`: seats on top of `tomo_motor` and takes the sample away from axis of
        rotation along horizontally-orthogonal axis to beam direction.
    - `align_motor_pbd`: seats on top of `tomo_motor` and takes the sample away from axis of
        rotation along horizontally-parallel axis to beam direction (i.e. in the same direction as
        the beam).
    - `rot_motor_pitch`: is the rotation motor which rotates around horizontally-orthogonal axis to
        the beam direction.
    - `rot_motor_roll`: is the rotation motor which rotates around horizontally-parallel axis to
        beam direction.

    Once the sample is brought into the FOV and been ensured that it remains in FOV across 360
    degrees of rotation, generally we want to first center the sample on rotation axis and
    projection as a preparatory step before off-centering for alignment. It provides us with a
    clean slate to start actual steps of alignment.
    
    STEP 0: This implementation assumes that sample is brought into FOV and can be rotated by 360
    degrees without sending it outside FOV. We can do this manually.

    STEP 1: Preload and initialize alignment state. Preloading is done as a prerequisite to
    backlash-compensated relative move of the motors implemented in the function `move_relative`,
    which describes the countermeasure against backlash.
    
    STEP 2: Centering is a two-step procedure, firstly we center the sample on axis of rotation
    and then we center the rotation stage w.r.t. projection. Axis of rotation is the geometric
    center of the tomographic rotation motor. Motor `align_motor_obd` takes the sample away from
    center in the horizontally-orthogonal direction to the beam. This offset distance then becomes
    the radius w.r.t. the cartesian axis along which this motor moves. Motor `align_motor_pbd` does
    the same for horizontally-parallel direction the beam. The choice of not using X or Y axis in
    concrete terms is deliberate, because these axes also rotate with the rotary stage.
    
    To center the sample on rotation axis we need to compute the said offset for each direction and
    adjust respective motors accordingly. At the end of these centering we should not see any
    horizontal displacement of the sample upon rotation. This is implemented in the function
    `center_sample_on_axis`. After centering the sample on axis of rotation we want to center the
    stage with respect to projection. It is done based on center of mass and implemented in
    `center_tomo_stage_in_projection`.

    Misalignment occurs from a very small tilt of the rotation plane either in the direction
    orthogonal to beam (roll error) and/or parallel to beam (pitch error). Projection being our only
    source of information we try to model this error using an imaginary right triangle on the
    projection plane. Let's say we are trying to estimate the tilt which is orthogonal to beam
    i.e. roll angle misalignment. To estimate this angle we need some vertical and horizontal
    displacements. While perfectly centered on axis of rotation sample shows no displacement upon
    rotating. Hence, we off-center the sample orthogonal to beam horizontally, which gives us some
    non-zero rotation radius and when we double this value it gives us the total horizontal shift
    in pixels for a 180 degree rotation. Furthermore, because of the roll angle error stage is not
    perfectly horizontal hence we get to see some vertical shifts in pixels for the same rotation.
    We get these shifts for normalized cross-correlation of two projection. Horizontal and vertical
    displacements measured in this way becomes the 'opposite' and 'adjacent' of our right triangle.
    Now we can measure the angular error, which caused the vertical shift using arctangent function.
    The same idea applies to the angular error parallel to beam direction (pitch) where we perform
    similar measurements after rotating the same 90 degrees, which is the only way to see in
    projections, the displacements made parallel to beam direction.

    Misalignments of roll and pitch angles cannot be reliably corrected in same iteration. Since
    our only source of information is projection we see the combined effect of both and upon
    estimating vertical offset of the sample their individual effects interfere with each other. In
    presence of both roll and pitch angle misalignments, it is quite possible that estimated
    vertical sample offset between [0, 180] degrees of rotation and [90, 270] degrees of rotation
    are same in magnitude and just reverse in sign. Therefore, we need to deal with angular errors
    in each direction individually.
    
    It is also required that we re-center the sample in projection after each iteration of angular
    correction. Since we off-center to estimate the error, making even a small angular adjustment
    to axial rotation motors in off-centered situation may cause the sample go outside FOV.

    STEP 3: Off-center the sample orthogonal to beam direction and iteratively correct roll angle
    misalignment.

    NOTE: After roll angle adjustment we move sample back to rotation axis and reset tomo angle to
    0 degree. Since we centered the sample before starting the angular corrections now irrespective
    of any potential pitch angle error we expect no displacement upon rotation. But we need to
    off-center again, this time parallel to beam direction to estimate the pitch angle error because
    we need the 'opposite' and 'adjacent' values from vertical and horizontal offsets.

    To estimate these offsets two specific aspects are relevant.
    
    - We first need to off-center the sample using linear motor parallel to beam direction. If we
    now set tomo position to 90 degrees and cross-correlate projections taken at angular range of
    [90, 270] we would are expected to see only horizontal offset (proportional to off-centering)
    and vertical offset should be theoretically zero. In context of alignment we rotate tomo stage
    to 90 degrees to be able to see how far we are from axis along the direction parallel to beam.
    And when we rotate to 270 degrees position starting from 90 degrees position in absolute sense
    sample remains theoretically at the same height at start and end positions. Hence, from this
    angular range we can only estimate the 'adjacent' (horizontal offset from center) part.

    - To capture the 'opposite' (vertical offset) part, right after off-centering parallel to beam
    direction we need to cross-correlate the projections at angular range of [0, 180] degrees. This
    has the opposite situation to the [90, 270] degrees range. With [0, 180] degrees range, if there
    exist a pitch angle error we would only see the vertical offset of the sample positions at start
    and end of he rotation. In this case before and after the rotation sample remains theoretically
    at the same horizontal position.

    STEP 4: Off-center the sample parallel to beam direction and iteratively correct pitch (lamino)
    angle misalignment.

    :param acq_devices: devices, which are collectively required for acquisition
    :type acq_devices: `concert.processes.common.AcquisitionDevices`
    :param acq_params: configurations specific to acquisition
    :type acq_params: `concert.processes.common.AcquisitionParams`
    :param align_devices: devices which are collectively required for alignment
    :type align_devices: `concert.processes.common.AlignmentDevices`
    :param align_params: configurations specific to alignment
    :type align_params: `concert.processes.common.AlignmentParams`
    :param pixel_size_um: pixel size in microns
    :type pixel_size_um: `concert.quantities.Quantity`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later

    async def _preload(motor: Motor_T) -> None:
        """
        Ensures that gear of the `motor` touches the face on the +ve side to ensure accuracy of
        subsequent moves with backlash compensation.
        """
        bl_comp: Quantity = align_params.bl_comp_rel_rot if isinstance(motor, RotationMotor) \
        else align_params.bl_comp_rel_lin
        await motor.move(-2 * bl_comp)
        await motor.move(2 * bl_comp)        

    async def _get_roll() -> Quantity:
        """Estimates roll angle error"""
        ver_obd_px, hor_obd_px = await get_sample_shifts(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
            align_state=align_state, start_ang_deg=0 * q.deg)
        return np.rad2deg(np.arctan2(ver_obd_px, 2 * hor_obd_px)) * q.deg
    
    async def _step_roll(curr_roll: Quantity) -> Quantity:
        """Makes a corrective step for roll angle error"""
        delta_move: Quantity = np.sign(curr_roll) * align_params.delta_move_deg
        await align_devices.rot_motor_roll.move(delta_move)
        _roll: Quantity = await _get_roll()
        await align_devices.rot_motor_roll.move(-delta_move)
        if abs(_roll) > abs(curr_roll):
            curr_roll = -curr_roll
        # await align_devices.rot_motor_roll.move(curr_roll)
        await move_relative(
            motor=align_devices.rot_motor_roll, move_by=curr_roll,
            align_params=align_params, logger=logger)
        return await _get_roll()
    
    async def _get_pitch() -> Quantity:
        """Estimates pitch angle error"""
        ver_pbd_px, _ = await get_sample_shifts(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
            align_state=align_state, start_ang_deg=0 * q.deg)
        _, hor_pbd_px = await get_sample_shifts(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
            align_state=align_state, start_ang_deg=90 * q.deg)
        return np.rad2deg(np.arctan2(ver_pbd_px, 2 * hor_pbd_px)) * q.deg
    
    async def _step_pitch(curr_pitch: Quantity) -> Quantity:
        """Makes a corrective step for pitch angle error"""
        delta_move: Quantity = np.sign(curr_pitch) * align_params.delta_move_deg
        await align_devices.rot_motor_pitch.move(delta_move)
        _pitch: Quantity = await _get_pitch()
        await align_devices.rot_motor_pitch.move(-delta_move)
        if abs(_pitch) > abs(curr_pitch):
            curr_pitch = -curr_pitch
        # await align_devices.rot_motor_pitch.move(curr_pitch)
        await move_relative(
            motor=align_devices.rot_motor_pitch, move_by=curr_pitch,
            align_params=align_params, logger=logger)
        return await _get_pitch()

    func_name = "align_rotation_stage_comparative"
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    logger.debug(f"{func_name}")
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    # STEP 0: Sample is brought into FOV and a 360 degrees rotation is possible without sending it
    # outside FOV.
    # STEP 1: Preload and initialize alignment state.
    logger.debug("Start: preload for backlash compensation.")
    await _preload(align_devices.align_motor_obd)
    await _preload(align_devices.align_motor_pbd)
    await _preload(align_devices.rot_motor_roll)
    await _preload(align_devices.rot_motor_pitch)
    logger.debug("Done: preload.")
    logger.debug("Start: state initialization before alignment.")
    align_state: AlignmentState = await init_alignment_state(
        acq_devices=acq_devices, acq_params=acq_params, align_devices=align_devices)
    logger.debug(f"Done: state initialized as:\n{align_state}")
    # STEP 2: Center sample into projection in two steps, firstly centering the sample to the axis
    # and then centering the tomo stage to the projection by center of mass.
    logger.debug("Start: centering sample on axis.")
    await center_sample_on_axis(
        acq_devices=acq_devices, acq_params=acq_params,
        align_devices=align_devices, align_params=align_params,
        align_state=align_state, pixel_size_um=pixel_size_um)
    logger.debug("Done: sample centered on axis.")
    logger.debug("Start: centering sample in projection.")
    await center_stage_in_projection(
        acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
        pixel_size_um=pixel_size_um)
    logger.debug("Done: sample centered in projection.")
    # STEP 3: Off-center the sample orthogonal to beam direction and iteratively correct roll angle
    # misalignment.
    logger.debug("Start: alignment.")
    roll_iter = 0
    await align_devices.align_motor_obd.move(align_params.offset_obd)
    roll_ang_err: Quantity = await _get_roll()
    logger.debug(f"::roll before misalignment: {abs(roll_ang_err)}")
    while abs(roll_ang_err) > acq_params.align_metric:
        logger.debug(f"::roll-correction iteration: {roll_iter}")
        roll_ang_err = await _step_roll(curr_roll=roll_ang_err)
        await align_devices.align_motor_obd.move(-align_params.offset_obd)
        await center_stage_in_projection(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
            pixel_size_um=pixel_size_um)
        roll_iter += 1
    logger.debug(f"::roll after misalignment: {abs(roll_ang_err)}")
    # STEP 4: Off-center the sample parallel to beam direction and iteratively correct pitch
    # (lamino) angle misalignment.
    pitch_iter = 0
    await align_devices.align_motor_pbd.move(align_params.offset_pbd)
    pitch_ang_err: Quantity = await _get_pitch()
    logger.debug(f"::pitch before misalignment: {abs(pitch_ang_err)}")
    while abs(pitch_ang_err) > acq_params.align_metric:
        logger.debug(f"::pitch correction iteration: {pitch_iter}")
        pitch_ang_err = await _step_pitch(curr_pitch=pitch_ang_err)
        await align_devices.align_motor_pbd.move(-align_params.offset_pbd)
        await center_stage_in_projection(
            acq_devices=acq_devices, acq_params=acq_params, align_params=align_params,
            pixel_size_um=pixel_size_um)
        pitch_iter += 1
    logger.debug(f"::pitch after misalignment: {abs(pitch_ang_err)}")
    await acq_devices.tomo_motor.set_position(0 * q.deg)
    logger.debug("Done: alignment.")
####################################################################################################


class ProcessError(Exception):

    """
    Exception raised by a process when something goes wrong with the procedure
    it tries to accomplish, e.g. cannot focus, cannot align rotation axis, etc.

    """
    pass



