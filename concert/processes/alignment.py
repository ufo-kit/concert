"""
alignment.py
------------
Alignment routines for measurement stages.
"""
from dataclasses import dataclass
import logging
from typing import Callable, Tuple, Dict, Optional
import numpy as np
from numpy import ndarray
import skimage.feature as sft
import skimage.measure as sms
import skimage.draw as sdr
from skimage.measure._regionprops import RegionProperties
from concert.coroutines.base import background
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.devices.cameras.base import Camera
from concert.devices.shutters.base import Shutter
from concert.ext.viewers import PyQtGraphViewer
from concert.imageprocessing import flat_correct
from concert.processes.common import ProcessError
from concert.quantities import q, Quantity
from concert.typing import Motor_T


LOG = logging.getLogger(__name__)


def locate_template(frame: ndarray, patch: ndarray) -> Tuple[int, int, float]:
    """
    Return the template center (row, column) and maximum correlation score.

    :param frame: projection of the alignment phantom
    :type frame: ndarray
    :param patch: localized patch of the alignment phantom
    :type patch: ndarray
    :return: template center (row, column) and maximum correlation score
    :rtype: Tuple[int, int, float]
    """
    matched = sft.match_template(image=frame, template=patch, pad_input=True)
    row, column = np.unravel_index(np.argmax(matched), matched.shape)
    return int(row), int(column), float(np.max(matched))


def log_residual(
        logger: logging.Logger,
        stage: str,
        residual: Quantity,
        tolerance: Quantity,
        iterations: int,
        max_iterations: int) -> None:
    """
    Logs measured residual distance error at different stages of alignment workflow.

    :param logger: logger
    :type logger: logging.Logger
    :param stage: context for logging
    :type stage: str
    :param residual: residual distance error after max iterations are reached
    :type residual: `concert.quantities.Quantity`
    :param tolerance: distance tolerance for a motor
    :type tolerance: `concert.quantities.Quantity`
    :param iterations: current iteration of convergence
    :type iterations: int
    :param max_iterations: max iterations
    :type max_iterations: int
    """
    converged = bool(abs(residual) <= tolerance)
    logger.info("%s: residual=%s, tolerance=%s, iterations=%d, converged=%s",
                stage, abs(residual), tolerance, iterations, converged)
    if not converged and iterations >= max_iterations:
        logger.warning("%s: iteration limit reached above tolerance", stage)


class BacklashCompRelMovMixin:
    """
    Facilitates backlash-compensated relative movement for motors. When some specific routine to
    account for the backlash is not implemented in the controller, motors often suffer from some
    amount of potential backlash, leading to an imprecise movement. The amount of real backlash in
    high precision motors is hard to estimate. The default compensation distances are 0.1 mm for
    linear motors and 0.1 degrees for rotation motors, assuming the real backlash is smaller.
    Backlash takes place when some movement makes the motor to change direction from +ve to -ve
    or vice versa. We work with two subroutines, namely `preload` and `move`. For both routines our
    objective is to approach all final motor movements toward +ve direction and at the end of
    the movement align the motor position toward +ve side.

    Before making adjustments with any motors preload attempts to make sure that the motor is
    aligned towards the side of +ve movement by making a bigger move covering the backlash toward
    -ve direction and then comeback all the way toward +ve direction.

    Move assumes that the motor is preloaded toward +ve direction. When the required movement
    is +ve it makes the move without any adjustment. In the alternative situation it makes the
    -ve movement added with the compensatory amount in the same direction and then makes the
    compensatory move only from -ve to +ve direction to make sure that at the end of the movement
    the motor is again aligned toward +ve direction.

    - **bl_comp_lin**: relative movement distance to counter backlash for linear motors.
    - **bl_comp_rot**: relative movement distance to counter backlash for rotation motors.
    """

    bl_comp_lin: Quantity = 0.1 * q.mm
    bl_comp_rot: Quantity = 0.1 * q.deg

    async def preload(self, motor: Motor_T) -> None:
        """
        Ensures that gears of the `motor` touch the face on the +ve side to ensure accuracy of
        subsequent relative moves with backlash compensation.

        :param motor: linear or rotation motor used for alignment
        :type motor: `concert.typing.Motor_T`
        """
        bl_comp: Quantity = self.bl_comp_rot if isinstance(
            motor, RotationMotor) else self.bl_comp_lin
        await motor.move(-2 * bl_comp)
        await motor.move(2 * bl_comp)

    async def move(self, motor: Motor_T, distance: Quantity) -> None:
        """
        Makes a backlash-compensated relative movement by overshooting to the -ve direction and
        then approaching towards +ve direction.

        :param motor: linear or rotation motor used for alignment
        :type motor: `concert.typing.Motor_T`
        :param distance: relative distance to move the motor
        :type distance: `concert.quantities.Quantity`
        """
        if np.sign(distance) > 0:
            await motor.move(distance)
            return
        bl_comp: Quantity = self.bl_comp_rot if isinstance(
            motor, RotationMotor) else self.bl_comp_lin
        await motor.move(distance - bl_comp)
        await motor.move(bl_comp)


@dataclass
class AcquisitionDevices:
    """
    Encapsulates relevant devices which are collectively used to acquire frames from camera under
    given positions of the rotary stage.

    - **camera**: reference to camera.
    - **shutter**: reference to shutter.
    - **tomo_motor**: reference to tomographic rotation motor.
    - **flat_motor**: reference to linear motor moving tomographic rotation stage horizontally.
    - **z_motor**: reference to linear motor moving tomographic rotation stage vertically.
    """
    camera: Camera
    shutter: Shutter
    tomo_motor: RotationMotor
    flat_motor: LinearMotor
    z_motor: LinearMotor


@dataclass
class AcquisitionContext(BacklashCompRelMovMixin):
    """
    Encapsulates devices and configurations to acquire frames from camera under given positions of
    the rotary stage.

    - **devices**: reference to devices which are relevant for acquiring frames using camera.
    - **height**: height of the projections.
    - **width**: width of the projections.
    - **flat_field_correct**: if flat field correction should be done for the projections.
    - **absorptivity**: flag indicating if absorptivity needs to be calculated.
    - **flat_position**: optional position of the flat motor to move sample away from beam \
        (required when ``flat_field_correct`` is true).
    - **vert_crop_start**: inclusive first row, defaults to 0.
    - **vert_crop_end**: exclusive end row, defaults to None to include the final row. Explicit \
        negative indices follow Python slicing; -1 excludes the final row.
    """
    devices: AcquisitionDevices
    height: int
    width: int
    flat_field_correct: bool
    absorptivity: bool
    flat_position: Optional[Quantity] = None
    vert_crop_start: int = 0
    vert_crop_end: Optional[int] = None


@dataclass
class AlignmentDevices:
    """
    Encapsulates relevant devices for alignment for which we might need to make frequent small
    adjustments.

    - **rot_motor_pitch**: rotation motor for pitch angle correction.
    - **rot_motor_roll**: rotation motor for roll angle correction.
    - **align_motor_pbd**: linear alignment motor to move sample horizontally parallel to the beam.
    - **align_motor_obd**: linear alignment motor to move sample horizontally orthogonal to the \
        beam.
    """
    rot_motor_pitch: RotationMotor
    rot_motor_roll: RotationMotor
    align_motor_pbd: LinearMotor
    align_motor_obd: LinearMotor


@dataclass
class AlignmentContext(BacklashCompRelMovMixin):
    """
    Encapsulates devices and configurations for the alignment.

    - **devices**: reference to the devices, which are relevant for alignment of tomographic stage.
    - **pixel_size_um**: calibrated sample-plane pixel size as a length quantity.
    - **max_iterations**: max iterations for alignment.
    - **pixel_sensitivity**: pixel sensitivity to derive a metric to evaluate alignment.
    - **offset_tomo**: angular offset to be applied to tomographic rotation motor. \
        Default implementation assumes no offset and linear alignment motor, which translates \
        the sample orthogonal to the beam direction is oriented along the rotation range of \
        [0, 180] degrees. This property gives the freedom to alter the default behavior by applying
        and arbitrary rotation phase.
    - **off_cent_pbd**: off-centering distance for alignment motor moving parallel to beam.
    - **del_dist_lin**: linear delta distance to determine correct direction.
    - **del_dist_rot**: angular delta distance to determine correct direction.
    - **pixel_err_eps**: epsilon pixel error tolerance during centering the sample.
    - **adjust_move**: divisor for sample-centering corrections (experimental); values above \
        one reduce the correction distance.
    - **proc_func**: image processing function to separate sphere from background.
    - **viewer**: optional viewer for the initial patch mosaic and alignment frames. No viewer \
        is created by the algorithm.
    """
    devices: AlignmentDevices
    pixel_size_um: Quantity
    max_iterations: int = 10
    pixel_sensitivity: int = 2
    offset_tomo: Quantity = 0 * q.deg
    off_cent_pbd: Quantity = 2 * q.mm
    del_dist_lin: Quantity = 0.1 * q.mm
    del_dist_rot: Quantity = 0.05 * q.deg
    pixel_err_eps: float = 2.0
    adjust_move: float = 1.0
    proc_func: Callable[[ndarray], ndarray] = lambda x: x
    viewer: Optional[PyQtGraphViewer] = None


@dataclass
class AlignmentState:
    """
    Encapsulates elements of the state management for the alignment.

    - **checkpoints**: motor positions recorded at initialization; not updated or used for recovery.
    - **patches**: contains a patch of our sample for each of the terminal angles.
    - **baseline_scores**: confidence scores for the sample being inside FOV.
    - **dark**: optional cached dark field.
    - **flat**: optional cached flat-field.
    - **sphere_radius**: sphere radius, to be derived during state initialization.
    - **dim**: patch dimension.
    - **score_epsilon**: maximum uncertainty to allow to conclude that the sample is in fact \
        inside FOV.

    NOTE: The idea for the checkpoints dictionary is to track the last known 'good' motor positions
    for which sample was definitely in FOV. It is supposed to eventually serve state management
    during alignment and help in recovering from anomalies like sample going outside FOV.
    Implementation for checkpoints-based recovery system is still in planning. Until it is
    implemented alignment relies on sample strictly remaining inside FOV during the process.
    """
    checkpoints: Dict[str, Quantity]
    patches: Dict[str, ndarray]
    baseline_scores: Dict[str, float]
    dark: Optional[ndarray] = None
    flat: Optional[ndarray] = None
    sphere_radius: Optional[int] = None
    dim: int = 200
    score_epsilon: float = 0.2

    def __str__(self) -> str:
        "Provides a printable expression for the state"
        val = "Motor Positions:\n"
        for key, value in self.checkpoints.items():
            val += f" {key} = {value}\n"
        val += "Baseline Scores (expected close to 1.0):\n"
        for key, value in self.baseline_scores.items():
            val += f" {key} degree = {value}\n"
        val += f"Sphere Radius = {self.sphere_radius}"
        return val

    def sample_in_FOV(self, frame: ndarray, angle: int) -> bool:
        """
        Evaluates if sample is inside FOV for the given `frame` by evaluating the
        confidence score against baseline score for given `angle`.

        :param frame: projection to evaluate
        :type frame: `numpy.ndarray`
        :param angle: angle to select the patch and score
        :type angle: int
        :return: if sample inside FOV
        :rtype: bool
        """
        _, _, score = locate_template(frame, self.patches[str(angle)])
        return abs(self.baseline_scores[str(angle)] - abs(score)) < self.score_epsilon


async def acquire_frame(acq_ctx: AcquisitionContext, align_state: AlignmentState) -> ndarray:
    """
    Acquires a single frame using context provided for acquisition.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.alignment.AcquisitionContext`
    :param align_state: state for alignment
    :type align_state: `concert.processes.alignment.AlignmentState`
    :return: acquired frame
    :rtype: `numpy.ndarray`
    """
    frame: ndarray = await acq_ctx.devices.camera.grab()
    if acq_ctx.flat_field_correct:
        if align_state.dark is None:
            if await acq_ctx.devices.shutter.get_state() != 'closed':
                await acq_ctx.devices.shutter.close()
            align_state.dark = await acq_ctx.devices.camera.grab()
        if await acq_ctx.devices.shutter.get_state() != 'open':
            await acq_ctx.devices.shutter.open()
        if align_state.flat is None:
            radio_pose: Quantity = await acq_ctx.devices.flat_motor.get_position()
            await acq_ctx.devices.flat_motor.set_position(acq_ctx.flat_position)
            align_state.flat = await acq_ctx.devices.camera.grab()
            await acq_ctx.devices.flat_motor.set_position(radio_pose)
        frame = flat_correct(radio=frame, flat=align_state.flat, dark=align_state.dark)
    if acq_ctx.absorptivity:
        frame = np.nan_to_num(-np.log(frame))
    return np.asarray(frame)[acq_ctx.vert_crop_start:acq_ctx.vert_crop_end, :]


async def init_alignment_state(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        logger: logging.Logger = LOG) -> AlignmentState:
    """
    Initializes alignment state.

    - Record first checkpoint with relevant motor positions before alignment.
    - For each terminal angle, grab a frame,
        - extract a patch containing the sample,
        - derive sphere radius,
        - use template matching to get a baseline confidence score,
    - If a viewer is supplied, show all four patches in one square mosaic.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.alignment.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.alignment.AlignmentContext`
    :param logger: logger, defaults to the module logger
    :type logger: logging.Logger
    :return: initial alignment state
    :rtype: `concert.processes.alignment.AlignmentState`
    """
    def _patch_mosaic(patches: Dict[str, ndarray]) -> ndarray:
        """Make a padded square preview with 0/90 degrees above 180/270 degrees"""
        angles = ("0", "90", "180", "270")
        side = max(max(patches[angle].shape) for angle in angles)
        preview = np.zeros((2 * side, 2 * side),
                        dtype=np.result_type(*(patches[angle].dtype for angle in angles)))
        for index, angle in enumerate(angles):
            patch = patches[angle]
            height, width = patch.shape
            row = (index // 2) * side + (side - height) // 2
            column = (index % 2) * side + (side - width) // 2
            preview[row:row + height, column:column + width] = patch
        return preview

    logger.info("Start: state initialization before alignment.")
    state = AlignmentState(checkpoints={}, patches={}, baseline_scores={})
    # Record all relevant motor positions
    for motor_str in ["flat_motor", "z_motor"]:
        state.checkpoints[motor_str] = await getattr(acq_ctx.devices, motor_str).get_position()
    for motor_str in ["align_motor_obd", "align_motor_pbd", "rot_motor_roll", "rot_motor_pitch"]:
        state.checkpoints[motor_str] = await getattr(align_ctx.devices, motor_str).get_position()
    # Record patches and baseline scores for the terminal angles
    try:
        for angle in [0, 90, 180, 270]:
            await acq_ctx.devices.tomo_motor.set_position(angle * q.deg + align_ctx.offset_tomo)
            frame: ndarray = await acquire_frame(acq_ctx=acq_ctx, align_state=state)
            mask: ndarray = align_ctx.proc_func(frame)
            region: RegionProperties = sorted(
                sms.regionprops(label_image=sms.label(mask)),
                key=lambda r: r.eccentricity)[0]
            cnt_y, cnt_x = int(region.centroid[0]), int(region.centroid[1])
            dim = state.dim // 2
            state.patches[str(angle)] = frame[cnt_y - dim:cnt_y + dim, cnt_x - dim:cnt_x + dim]
            if not state.sphere_radius:
                state.sphere_radius = int(region.perimeter / (2 * np.pi))
            _, _, state.baseline_scores[str(angle)] = locate_template(
                frame, state.patches[str(angle)])
        if align_ctx.viewer:
            await align_ctx.viewer.set_title("Alignment patches: 0° | 90° / 180° | 270°")
            await align_ctx.viewer.show(_patch_mosaic(state.patches))
    except Exception:
        raise ProcessError("review proc_func and ensure 360 rotation is possible within FOV")
    finally:
        await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_tomo)
    logger.debug(f"Done: state initialized as:\n{state}")
    logger.info("State initialization finished.")
    return state


async def get_sample_shifts(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        tomo_angle: Quantity) -> Tuple[float, float]:
    """
    Estimates the vertical and horizontal shifts of the sample across a 180 degrees rotation. The
    parameter `tomo_angle` marks the starting angle before rotation.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.alignment.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.alignment.AlignmentContext`
    :param align_state: alignment state
    :type align_state: `concert.processes.alignment.AlignmentState`
    :param tomo_angle: initial angle(degrees) to set before measuring offset
    :type tomo_angle: `concert.quantities.Quantity`
    :return: full absolute vertical separation and half the absolute horizontal separation,
        both in pixels
    :rtype: Tuple[float, float]
    """

    async def _shifts(
            ref_img: ndarray, mov_img: ndarray, tomo_angle: int) -> Tuple[float, float]:
        """Locate the angle-specific templates and derive unsigned shifts in pixels."""
        ref_y, ref_x, _ = locate_template(ref_img, align_state.patches[str(tomo_angle)])
        mov_y, mov_x, _ = locate_template(mov_img, align_state.patches[str(tomo_angle + 180)])
        ver_shift, hor_shift = abs(mov_y - ref_y), abs(mov_x - ref_x) / 2
        if align_ctx.viewer:
            vis_frame: ndarray = ref_img + mov_img
            ref_rows, ref_cols = sdr.disk(center=(ref_y, ref_x), radius=12)
            move_rows, move_cols = sdr.disk(center=(mov_y, mov_x), radius=12)
            vis_frame[ref_rows, ref_cols] = np.min(vis_frame)
            vis_frame[move_rows, move_cols] = np.min(vis_frame)
            await align_ctx.viewer.show(vis_frame)
            await align_ctx.viewer.set_title(f"ver_shift = {ver_shift} hor_shift={hor_shift}")
        return ver_shift, hor_shift

    await acq_ctx.devices.tomo_motor.set_position(tomo_angle + align_ctx.offset_tomo)
    ref_img: ndarray = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
    if not align_state.sample_in_FOV(frame=ref_img, angle=tomo_angle.magnitude):
        raise ProcessError("sample went outside FOV, aborting")
    await acq_ctx.devices.tomo_motor.move(180 * q.deg)
    mov_img: ndarray = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
    if not align_state.sample_in_FOV(frame=mov_img, angle=tomo_angle.magnitude + 180):
        raise ProcessError("sample went outside FOV, aborting")
    await acq_ctx.devices.tomo_motor.move(-180 * q.deg)
    return await _shifts(ref_img=ref_img, mov_img=mov_img, tomo_angle=tomo_angle.magnitude)


async def center_sample_on_axis(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        logger: logging.Logger = LOG) -> None:
    """
    Adjusts alignment motors orthogonal to the beam direction, `align_motor_obd` and parallel
    to the beam direction, `align_motor_pbd` to center the sample on rotation axis.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.alignment.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.alignment.AlignmentContext`
    :param align_state: alignment state
    :type align_state: `concert.processes.alignment.AlignmentState`
    :param logger: logger, defaults to the module logger
    :type logger: logging.Logger
    """
    async def _step_center(motor: LinearMotor, curr_offset: float, tomo_angle: Quantity) -> float:
        """Makes one step of linear motor adjustment toward center of rotation"""
        await align_ctx.move(motor=motor, distance=align_ctx.del_dist_lin)
        _, _interim_offset = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=tomo_angle)
        await align_ctx.move(motor=motor, distance=-align_ctx.del_dist_lin)
        if _interim_offset > curr_offset:
            curr_offset = -curr_offset
        # TODO: Use of align_ctx.adjust_move is experimental. It is hard to configure this value
        # correctly. As an alternative we could try to derive the motor movement from computed
        # offset and align_ctx.max_iterations e.g., (curr_offset / align_ctx.max_iterations).
        await align_ctx.move(
            motor=motor, distance=(curr_offset / align_ctx.adjust_move) * align_ctx.pixel_size_um)
        _, _new_offset = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=tomo_angle)
        return _new_offset

    # Make step adjustment for alignment motor orthogonal to beam direction.
    _, offset_obd = await get_sample_shifts(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=0 * q.deg)
    logger.debug(f">> before: offset_obd = {offset_obd}")
    obd_iter = 0
    while offset_obd > align_ctx.pixel_err_eps and obd_iter < align_ctx.max_iterations:
        offset_obd = await _step_center(
            motor=align_ctx.devices.align_motor_obd, curr_offset=offset_obd, tomo_angle=0 * q.deg)
        logger.debug(f">>>> centering-obd iter = {obd_iter} offset_obd = {offset_obd}")
        obd_iter += 1
    log_residual(logger, "Sample centering OBD", offset_obd * q.px,
                  align_ctx.pixel_err_eps * q.px, obd_iter, align_ctx.max_iterations)

    # Make step adjustment for alignment motor parallel to beam direction.
    _, offset_pbd = await get_sample_shifts(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=90 * q.deg)
    logger.debug(f">> before: offset_pbd = {offset_pbd}")
    pbd_iter = 0
    while offset_pbd > align_ctx.pixel_err_eps and pbd_iter < align_ctx.max_iterations:
        offset_pbd = await _step_center(
            motor=align_ctx.devices.align_motor_pbd, curr_offset=offset_pbd, tomo_angle=90 * q.deg)
        logger.debug(f">>>> centering-pbd iter = {pbd_iter} offset_pbd = {offset_pbd}")
        pbd_iter += 1
    log_residual(logger, "Sample centering PBD", offset_pbd * q.px,
                  align_ctx.pixel_err_eps * q.px, pbd_iter, align_ctx.max_iterations)


async def offset_from_projection_center(
        acq_ctx: AcquisitionContext,
        align_state: AlignmentState) -> Tuple[float, float]:
    """
    Derives the distance in pixels between geometric center of the projection and the
    sample placed at the axis of rotation.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.alignment.AcquisitionContext`
    :param align_state: managed state for alignment
    :type align_state: `concert.processes.alignment.AlignmentState`
    :return: absolute vertical and horizontal distances in pixels between the projection center
        and template-localized sphere position; assumes the reference tomography angle
    :rtype: Tuple[float, float]
    """
    frame: ndarray = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
    patch = align_state.patches["0"]
    sample_y, sample_x, _ = locate_template(frame, patch)
    yc_proj, xc_proj = frame.shape[0] / 2, frame.shape[1] / 2
    z_offset, stage_offset = abs(yc_proj - sample_y), abs(xc_proj - sample_x)
    return z_offset, stage_offset


async def center_axis_in_projection(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        logger: logging.Logger = LOG) -> None:
    """
    Adjusts the vertical `z_motor` and horizontal stage `flat_motor` to put the rotation axis
    in the middle of the projection.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.alignment.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.alignment.AlignmentContext`
    :param align_state: managed state for alignment
    :type align_state: `concert.processes.alignment.AlignmentState`
    :param logger: logger, defaults to the module logger
    :type logger: logging.Logger
    """
    offset_types = {"z_offset": 0, "stage_offset": 1}

    async def _step_center(motor: LinearMotor, curr_offset: float, offset_type: str) -> float:
        """Makes one step of linear motor adjustment toward center of projection"""
        await acq_ctx.move(motor=motor, distance=align_ctx.del_dist_lin)
        _interim_offset = (await offset_from_projection_center(
            acq_ctx=acq_ctx, align_state=align_state))[offset_types[offset_type]]
        await acq_ctx.move(motor=motor, distance=-align_ctx.del_dist_lin)
        if _interim_offset > curr_offset:
            curr_offset = -curr_offset
        await acq_ctx.move(motor=motor, distance=curr_offset * align_ctx.pixel_size_um)
        if align_ctx.viewer:
            frame: ndarray = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
            await align_ctx.viewer.show(frame)
        return (await offset_from_projection_center(
            acq_ctx=acq_ctx, align_state=align_state))[offset_types[offset_type]]

    await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_tomo)
    # Make step adjustment for z-motor (vertically orthogonal to beam direction)
    z_offset, _ = await offset_from_projection_center(acq_ctx=acq_ctx, align_state=align_state)
    logger.debug(f">> before: z_offset = {z_offset}")
    z_offset_iter = 0
    while z_offset > align_ctx.pixel_err_eps and z_offset_iter < align_ctx.max_iterations:
        z_offset = await _step_center(
            motor=acq_ctx.devices.z_motor, curr_offset=z_offset, offset_type="z_offset")
        z_offset_iter += 1
        logger.debug(f">>>> z iter = {z_offset_iter} z_offset = {z_offset}")
    log_residual(logger, "Projection centering vertical", z_offset * q.px,
                  align_ctx.pixel_err_eps * q.px, z_offset_iter, align_ctx.max_iterations)

    # Make step adjustment for stage (flat) motor (horizontally orthogonal to beam direction)
    _, stage_offset = await offset_from_projection_center(acq_ctx=acq_ctx, align_state=align_state)
    logger.debug(f">> before: stage_offset = {stage_offset}")
    stage_offset_iter = 0
    while stage_offset > align_ctx.pixel_err_eps and stage_offset_iter < align_ctx.max_iterations:
        stage_offset = await _step_center(
            motor=acq_ctx.devices.flat_motor, curr_offset=stage_offset, offset_type="stage_offset")
        stage_offset_iter += 1
        logger.debug(f">>>> stage iter = {stage_offset_iter} stage_offset = {stage_offset}")
    log_residual(logger, "Projection centering horizontal", stage_offset * q.px,
                  align_ctx.pixel_err_eps * q.px, stage_offset_iter, align_ctx.max_iterations)


@background
async def align_tomo_stage_parallel_beam(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        logger: logging.Logger = LOG) -> None:
    """
    Runs sample centering and sequential pitch/roll correction for parallel-beam CT geometry.

    Initialize ``align_state`` separately before calling. Each stage logs its residual and
    convergence status. Reaching an iteration cap does not raise or stop subsequent stages;
    returning None does not guarantee that all tolerances were met.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.alignment.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.alignment.AlignmentContext`
    :param align_state: state managed for alignment
    :type align_state: `concert.processes.alignment.AlignmentState`
    :param logger: logger, defaults to the module logger
    :type logger: logging.Logger
    """
    def _flush_flat_fields() -> None:
        """Force acquiring new flat fields"""
        align_state.dark = None
        align_state.flat = None

    async def _get_ang_err(off_cent_px: float) -> Quantity:
        """Derives the angular error from offsets"""
        vertical_sample_shift, _ = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=0 * q.deg)
        return np.rad2deg(np.arctan2(vertical_sample_shift, 2 * off_cent_px)) * q.deg

    async def _step_ang_err(
            rot_motor: RotationMotor,
            off_cent_px: float,
            curr_err: Quantity) -> Quantity:
        """Makes one step correction for the angular error"""
        del_ang: Quantity = np.sign(curr_err) * align_ctx.del_dist_rot
        await align_ctx.move(motor=rot_motor, distance=del_ang)
        _interim_err: Quantity = await _get_ang_err(off_cent_px=off_cent_px)
        await align_ctx.move(motor=rot_motor, distance=-del_ang)
        logger.debug(f"Current = {curr_err} Interim Err = {_interim_err}")
        if abs(_interim_err) > abs(curr_err):
            curr_err = -curr_err
        logger.debug(f"Effective Correction = {curr_err}")
        await align_ctx.move(motor=rot_motor, distance=curr_err)
        logger.debug(f"After Correction Motor Position = {await rot_motor.get_position()}")
        return await _get_ang_err(off_cent_px=off_cent_px)

    logger.info("Start: tomography alignment procedure.")
    # Preload for backlash-compensated movement.
    logger.info("Start: preload for backlash compensation.")
    await acq_ctx.preload(acq_ctx.devices.flat_motor)
    await acq_ctx.preload(acq_ctx.devices.z_motor)
    await align_ctx.preload(align_ctx.devices.align_motor_obd)
    await align_ctx.preload(align_ctx.devices.align_motor_pbd)
    await align_ctx.preload(align_ctx.devices.rot_motor_roll)
    await align_ctx.preload(align_ctx.devices.rot_motor_pitch)
    logger.info("Done: preload.")
    # Bring sample onto the center of rotation.
    logger.info("Start: centering sample on axis.")
    await center_sample_on_axis(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, logger=logger)
    logger.info("Sample centering stage finished.")
    # Bring center of rotation to the center of projection. This is a prerequisite to deriving
    # the offset distance for roll correction.
    logger.info("Start: centering axis in projection.")
    await center_axis_in_projection(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, logger=logger)
    logger.info("Projection centering stage finished.")
    logger.info("Start: alignment.")
    # Geometric angular threshold: allowed vertical pixel displacement across projection width.
    # This criterion does not establish reconstruction resolution.
    metric: Quantity = np.rad2deg(np.arctan(align_ctx.pixel_sensitivity / acq_ctx.width)) * q.deg
    logger.debug(f"Calculated: alignment metric: {metric}")
    # Off-center the sample parallel to beam direction and iteratively correct pitch angle
    # misalignment.
    _flush_flat_fields()
    await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_tomo)
    logger.info("Start: off-centering for pitch correction.")
    off_cent_px_pitch: float = (align_ctx.off_cent_pbd.to(q.um) / align_ctx.pixel_size_um).magnitude
    await align_ctx.move(
        motor=align_ctx.devices.align_motor_pbd,
        distance=align_ctx.off_cent_pbd)
    pitch_ang_err: Quantity = await _get_ang_err(off_cent_px=off_cent_px_pitch)
    logger.debug(f"::pitch before iterations: {abs(pitch_ang_err)}")
    pitch_iter = 0
    while abs(pitch_ang_err) > metric and pitch_iter < align_ctx.max_iterations:
        pitch_ang_err = await _step_ang_err(
            rot_motor=align_ctx.devices.rot_motor_pitch,
            off_cent_px=off_cent_px_pitch,
            curr_err=pitch_ang_err)
        pitch_iter += 1
        logger.debug(f"::::pitch-correction iteration: {pitch_iter} pitch: {abs(pitch_ang_err)}")
    log_residual(logger, "Pitch correction", pitch_ang_err, metric,
                  pitch_iter, align_ctx.max_iterations)
    await align_ctx.move(
        motor=align_ctx.devices.align_motor_pbd,
        distance=-align_ctx.off_cent_pbd)
    logger.info("Done: came back from off-centering for pitch correction.")
    # Off-center the sample orthogonal to beam direction and iteratively correct roll angle
    # misalignment. Off-centering for roll is towards the right edge of the projection short by
    # twice of sphere diameter.
    # NOTE: Understand, if it is necessary to bring the sample back to rotation axis and center
    # axis to projection before calculating roll angle for each iteration. It is possible because
    # in the off-centered state making roll angle adjustments may send the sample outside FOV and
    # then immediate next roll angle estimation will fail.
    _flush_flat_fields()
    await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_tomo)
    logger.info("Start: off-centering for roll correction.")
    off_cent_px_roll: float = (acq_ctx.width // 2) - (4 * align_state.sphere_radius)
    await align_ctx.move(
        motor=align_ctx.devices.align_motor_obd,
        distance=off_cent_px_roll * align_ctx.pixel_size_um)
    roll_ang_err: Quantity = await _get_ang_err(off_cent_px=off_cent_px_roll)
    logger.debug(f"::roll before iterations: {abs(roll_ang_err)}")
    roll_iter = 0
    while abs(roll_ang_err) > metric and roll_iter < align_ctx.max_iterations:
        roll_ang_err = await _step_ang_err(
            rot_motor=align_ctx.devices.rot_motor_roll,
            off_cent_px=off_cent_px_roll,
            curr_err=roll_ang_err)
        roll_iter += 1
        logger.debug(f"::::roll-correction iteration: {roll_iter} roll: {abs(roll_ang_err)}")
    log_residual(logger, "Roll correction", roll_ang_err, metric,
                  roll_iter, align_ctx.max_iterations)
    await align_ctx.move(
        motor=align_ctx.devices.align_motor_obd,
        distance=-off_cent_px_roll * align_ctx.pixel_size_um)
    logger.info("Done: came back from off-centering for roll correction.")
    logger.info("Start: centering sample in projection.")
    await center_axis_in_projection(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, logger=logger)
    logger.info("Final projection centering stage finished.")
    logger.info("Tomography alignment procedure finished; see stage residuals.")
