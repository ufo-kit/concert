from dataclasses import dataclass
import logging
from typing import Callable, Tuple, Dict, Optional
import numpy as np
import skimage.feature as sft
import skimage.measure as sms
import skimage.draw as sdr
from skimage.measure._regionprops import RegionProperties
import skimage.registration as skr
from concert.coroutines.base import background
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.devices.cameras.base import Camera
from concert.devices.shutters.base import Shutter
from concert.ext.viewers import PyQtGraphViewer
from concert.imageprocessing import flat_correct
from concert.processes.common import ProcessError
from concert.quantities import q, Quantity
from concert.typing import ArrayLike, Motor_T


class BacklashCompRelMovMixin:
    """
    Facilitates backlash-compensated relative movement for motors.

    - relative movement distance to counter backlash for linear motors.
    - relative movement distance to counter backlash for rotation motors.
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
    Encapsulates relevant devices which are collectively used for frame acquisition.

    - reference to camera.
    - reference to shutter.
    - reference to tomographic rotation motor.
    - reference to linear motor moving tomographic rotation stage horizontally.
    - reference to linear motor moving tomographic rotation stage vertically.
    """
    camera: Camera
    shutter: Shutter
    tomo_motor: RotationMotor
    flat_motor: LinearMotor
    z_motor: LinearMotor


@dataclass
class AcquisitionContext(BacklashCompRelMovMixin):
    """
    Encapsulates devices and configurations for frame acquisition.

    - reference to devices which are relevant for acquiring frames using camera.
    - height of the projections.
    - width of the projections.
    - flag indicating if flat field correction should be done for the acquired frames.
    - flag indicating if absorptivity needs to ve calculated.
    - optional position of the flat motor to move sample away from beam (only relevant if \
    `flat_field_correct` is true).
    """
    devices: AcquisitionDevices
    height: int
    width: int
    flat_field_correct: bool
    absorptivity: bool
    flat_position: Optional[Quantity] = None


@dataclass
class AlignmentDevices:
    """
    Encapsulates relevant devices for alignment fo which we might need to make frequent small
    adjustments.

    - rotation motor for pitch angle correction.
    - rotation motor for roll angle correction.
    - linear alignment motor to move sample horizontally parallel to the beam.
    - linear alignment motor to move sample horizontally orthogonal to the beam.
    """
    rot_motor_pitch: RotationMotor
    rot_motor_roll: RotationMotor
    align_motor_pbd: Optional[LinearMotor]
    align_motor_obd: Optional[LinearMotor]


@dataclass
class AlignmentContext(BacklashCompRelMovMixin):
    """
    Encapsulates devices and configurations for the alignment method.

    - reference to the devices, which are relevant for alignment of tomographic stage.
    - pixel size in micrometer.
    - max iterations for alignment.
    - pixel sensitivity to derive a metric to evaluate alignment.
    - angular offset to be applied to tomographic rotation motor, defaults to no offset.
    - off-centering distance for alignment motor moving parallel to beam.
    - linear delta distance to determine correct direction.
    - angular delta distance to determine correct direction.
    - epsilon pixel error tolerance during centering the sample.
    - proportional adjustment to be made before moving motors (experimental).
    - image processing function to separate sphere from background.
    - optional viewer to display frames for debugging.
    - method to derive vertical and horizontal shifts, one of \
    ["phase_cross_corr", "template_match"].
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
    proc_func: Callable[[ArrayLike], ArrayLike] = lambda x: x
    viewer: Optional[PyQtGraphViewer] = None
    offset_method: str = "template_match"

    TEMPLATE_MATCH: str = "template_match"
    PHASE_CROSS_CORR: str = "phase_cross_corr"


def get_noop_logger() -> logging.Logger:
    """Provides a no-op logger"""
    logger = logging.getLogger("no-op")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@dataclass
class AlignmentState:
    """
    Encapsulates elements of the state management for the alignment.

    - checkpoints - last known motor positions for which sample was in FOV.
    - patches - contains a patch of our sample for each of the terminal angles.
    - baseline_scores - confidence scores for the sample being inside FOV.
    - optional cached dark field.
    - optional cached flat-field.
    - sphere radius, to be derived during state initialization.
    - patch dimension.
    - score_epsilon is the maximum uncertainty to allow to conclude that the sample is in \
    fact inside FOV.

    TODO: Checkpoint based system is not fully implemented yet. It is supposed to serve state
    management during alignment and help in recovering from anomalies like sample going outside FOV.

    The idea for the checkpoints dictionary is to track the last known 'good' motor positions for
    which sample was definitely in FOV.

    Additionally, we need a stack data structure which should maintain a history of the
    chronological motor movements. At any given point in time if we detect the sample outside FOV
    from `checkpoints` and `history` we can derive which motor movement caused it and try to recover
    from it.
    """
    checkpoints: Dict[str, Quantity]
    patches: Dict[str, ArrayLike]
    baseline_scores: Dict[str, float]
    dark: Optional[ArrayLike] = None
    flat: Optional[ArrayLike] = None
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

    def sample_in_FOV(self, frame: ArrayLike, angle: int) -> bool:
        """
        Evaluates if sample is inside FOV for the given `frame` by evaluating the
        confidence score against baseline score for given `angle`.

        :param frame: projection to evaluate
        :type frame: `concert.typing.ArrayLike`
        :param angle: angle to select the patch and score
        :type angle: int
        :return: if sample inside FOV
        :rtype: bool
        """
        score: float = np.max(
            sft.match_template(image=frame, template=self.patches[str(angle)], pad_input=True))
        return abs(self.baseline_scores[str(angle)] - abs(score)) < self.score_epsilon


async def acquire_frame(acq_ctx: AcquisitionContext, align_state: AlignmentState) -> ArrayLike:
    """
    Acquires a single frame using context provided for acquisition.

    :param ctx: context for acquisition
    :type ctx: `concert.processes.common.AcquisitionContext`
    :param align_state: state for alignment
    :type align_state: `concert.processes.common.AcquisitionState`
    :return: acquired frame
    :rtype: `concert.typing.ArrayLike`
    """
    frame: ArrayLike = await acq_ctx.devices.camera.grab()
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
    return np.asarray(frame)


async def init_alignment_state(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        logger: logging.Logger = get_noop_logger()) -> AlignmentState:
    """
    Initializes alignment state.

    - Record first checkpoint with relevant motor positions before alignment.
    - For each terminal angle, grab a frame,
        - extract a patch containing the sample,
        - derive sphere radius,
        - use template matching to get a baseline confidence score,
        - show extracted patch using runtime viewer as sanity check.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :return: initial alignment state
    :rtype: `concert.processes.common.AlignmentState`
    """
    logger.debug = print  # TODO: For quick debugging, remove later

    logger.debug("Start: state initialization before alignment.")
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
            frame: ArrayLike = await acquire_frame(acq_ctx=acq_ctx, align_state=state)
            mask: ArrayLike = align_ctx.proc_func(frame)
            region: RegionProperties = sorted(
                sms.regionprops(label_image=sms.label(mask)),
                key=lambda r: r.eccentricity)[0]
            cnt_y, cnt_x = int(region.centroid[0]), int(region.centroid[1])
            dim = state.dim // 2
            state.patches[str(angle)] = frame[cnt_y - dim:cnt_y + dim, cnt_x - dim:cnt_x + dim]
            if not state.sphere_radius:
                state.sphere_radius = int(region.perimeter / (2 * np.pi))
            state.baseline_scores[str(angle)] = np.max(
                sft.match_template(image=frame, template=state.patches[str(angle)], pad_input=True))
            viewer = await PyQtGraphViewer(show_refresh_rate=True)
            await viewer.set_title(f"Angle = {str(angle)}")
            await viewer.show(state.patches[str(angle)])
    except Exception:
        raise ProcessError("review proc_func and ensure 360 rotation is possible within FOV")
    finally:
        await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_tomo)
    logger.debug(f"Done: state initialized as:\n{state}")
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
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param align_state: alignment state
    :type align_state: `concert.processes.common.AlignmentState`
    :param tomo_angle: initial angle(degrees) to set before measuring offset
    :type tomo_angle: `concert.quantities.Quantity`
    :return: vertical shift of sample caused by misalignment and distance of sample from center
    of rotation
    :rtype: Tuple[float, float]
    """

    async def _shifts(
            ref_img: ArrayLike, mov_img: ArrayLike, tomo_angle: int) -> Tuple[float, float]:
        """
        Derives vertical and horizontal shifts using either phase_cross_correlation of
        frames or matching pre-recorded template patches on individual frames.
        """
        ver_shift, hor_shift = 0, 0
        vis_frame: ArrayLike = ref_img + mov_img
        if align_ctx.offset_method == align_ctx.PHASE_CROSS_CORR:
            shift_yx, _, _ = skr.phase_cross_correlation(
                reference_image=ref_img, moving_image=mov_img, upsample_factor=4)
            ver_shift, hor_shift = abs(shift_yx[0]), abs(shift_yx[1]) / 2
        if align_ctx.TEMPLATE_MATCH:
            ref_patch: ArrayLike = align_state.patches[str(tomo_angle)]
            mov_patch: ArrayLike = align_state.patches[str(tomo_angle + 180)]
            ref_match: ArrayLike = sft.match_template(
                image=ref_img, template=ref_patch, pad_input=True)
            mov_match: ArrayLike = sft.match_template(
                image=mov_img, template=mov_patch, pad_input=True)
            ref_ind: ArrayLike = np.unravel_index(np.argmax(ref_match), ref_match.shape)
            ref_x, ref_y = ref_ind[::-1]
            mov_ind: ArrayLike = np.unravel_index(np.argmax(mov_match), mov_match.shape)
            mov_x, mov_y = mov_ind[::-1]
            ver_shift, hor_shift =  abs(mov_y - ref_y), abs(mov_x - ref_x) / 2
            if align_ctx.viewer:
                ref_rows, ref_cols = sdr.disk(center=(ref_y, ref_x), radius=12)
                move_rows, move_cols = sdr.disk(center=(mov_y, mov_x), radius=12)
                vis_frame[ref_rows, ref_cols] = np.min(vis_frame)
                vis_frame[move_rows, move_cols] = np.min(vis_frame)
        if align_ctx.viewer:
            await align_ctx.viewer.show(vis_frame)
            await align_ctx.viewer.set_title(f"ver_shift = {abs(ver_shift)} hor_shift={hor_shift}")
        return ver_shift, hor_shift

    await acq_ctx.devices.tomo_motor.set_position(tomo_angle + align_ctx.offset_tomo)
    ref_img: ArrayLike = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
    if not align_state.sample_in_FOV(frame=ref_img, angle=tomo_angle.magnitude):
        raise ProcessError("sample went outside FOV, aborting")
    await acq_ctx.devices.tomo_motor.move(180 * q.deg)
    mov_img: ArrayLike = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
    if not align_state.sample_in_FOV(frame=mov_img, angle=tomo_angle.magnitude + 180):
        raise ProcessError("sample went outside FOV, aborting")
    await acq_ctx.devices.tomo_motor.move(-180 * q.deg)
    return await _shifts(ref_img=ref_img, mov_img=mov_img, tomo_angle=tomo_angle.magnitude)


async def center_sample_on_axis(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        logger: logging.Logger = get_noop_logger()) -> None:
    """
    Adjusts alignment motors orthogonal to the beam direction, `align_motor_obd` and parallel
    to the beam direction, `align_motor_pbd` to center the sample on rotation axis.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param align_state: alignment state
    :type align_state: `concert.processes.common.AlignmentState`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug = print  # TODO: For quick debugging, remove later

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
    logger.debug(f">> after: offset_obd = {offset_obd}")

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
    logger.debug(f">> after: offset_pbd = {offset_pbd}")


async def offset_from_projection_center(
        acq_ctx: AcquisitionContext,
        align_state: AlignmentState) -> Tuple[float, float]:
    """
    Derives the distance in pixels between geometric center of the projection and the
    sample placed at the axis of rotation.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_state: managed state for alignment
    :type align_state: `concert.processes.common.AlignmentContext`
    :param tomo_angle: angle of the tomo rotation motor
    :type tomo_angle: `concert.quantities.Quantity`
    :return: distance between the geometric center of projection and center of mass
    :rtype: Tuple[float, float]
    """
    frame: ArrayLike = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
    patch = align_state.patches["0"]
    matched = sft.match_template(image=frame, template=patch, pad_input=True)
    indices = np.unravel_index(np.argmax(matched), matched.shape)
    xcm, ycm = indices[::-1]
    yc_proj, xc_proj = frame.shape[0] / 2, frame.shape[1] / 2
    z_offset, stage_offset = abs(yc_proj - ycm), abs(xc_proj - xcm)
    return z_offset, stage_offset


async def center_axis_in_projection(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        logger: logging.Logger = get_noop_logger()) -> None:
    """
    Adjusts the vertical `z_motor` and horizontal stage `flat_motor` to put the rotation axis
    in the middle of the projection.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param align_state: managed state for alignment
    :type align_state: `concert.processes.common.AlignmentState`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug = print  # TODO: For quick debugging, remove later

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
            frame: ArrayLike = await acquire_frame(acq_ctx=acq_ctx, align_state=align_state)
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
    logger.debug(f">> after: z_offset = {z_offset}")

    # Make step adjustment for stage (flat) motor (horizontally orthogonal to beam direction)
    _, stage_offset = await offset_from_projection_center(acq_ctx=acq_ctx, align_state=align_state)
    logger.debug(f">> before: stage_offset = {stage_offset}")
    stage_offset_iter = 0
    while stage_offset > align_ctx.pixel_err_eps and stage_offset_iter < align_ctx.max_iterations:
        stage_offset = await _step_center(
            motor=acq_ctx.devices.flat_motor, curr_offset=stage_offset, offset_type="stage_offset")
        stage_offset_iter += 1
        logger.debug(f">>>> stage iter = {stage_offset_iter} stage_offset = {stage_offset}")
    logger.debug(f">> after: stage_offset = {stage_offset}")


@background
async def align_tomography_generic(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        logger: logging.Logger = get_noop_logger()) -> None:
    """
    Aligns rotation stage for parallel beam CT geometry.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param align_state: state managed for alignment
    :type align_state: `concert.processes.common.AlignmentState`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug = print  # TODO: For quick debugging, remove later

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

    func_name = "align_rotation_stage_comparative"
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    logger.debug(f"{func_name}")
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    # Preload for backlash-compensated movement.
    logger.debug("Start: preload for backlash compensation.")
    await acq_ctx.preload(acq_ctx.devices.flat_motor)
    await acq_ctx.preload(acq_ctx.devices.z_motor)
    await align_ctx.preload(align_ctx.devices.align_motor_obd)
    await align_ctx.preload(align_ctx.devices.align_motor_pbd)
    await align_ctx.preload(align_ctx.devices.rot_motor_roll)
    await align_ctx.preload(align_ctx.devices.rot_motor_pitch)
    logger.debug("Done: preload.")
    # Bring sample onto the center of rotation.
    logger.debug("Start: centering sample on axis.")
    await center_sample_on_axis(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, logger=logger)
    logger.debug("Done: sample centered on axis.")
    # Bring center of rotation to the center of projection. This is a prerequisite to deriving
    # the offset distance for roll correction.
    logger.debug("Start: centering axis in projection.")
    await center_axis_in_projection(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, logger=logger)
    logger.debug("Done: axis centered in projection.")
    logger.debug("Start: alignment.")
    # Alignment metric is a measure of resolution sensitivity. We derive it as an angular threshold,
    # which represents maximum #pixels of resolution loss vertically against projection width
    # pixels horizontally.
    metric: Quantity = np.rad2deg(np.arctan(align_ctx.pixel_sensitivity / acq_ctx.width)) * q.deg
    logger.debug(f"Calculated: alignment metric: {metric}")
    # Off-center the sample parallel to beam direction and iteratively correct pitch angle
    # misalignment.
    _flush_flat_fields()
    await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_tomo)
    logger.debug("Start: off-centering for pitch correction.")
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
    logger.debug(f"::pitch after iterations: {abs(pitch_ang_err)}")
    await align_ctx.move(
        motor=align_ctx.devices.align_motor_pbd,
        distance=-align_ctx.off_cent_pbd)
    logger.debug("Done: came back from off-centering for pitch correction.")
    # Off-center the sample orthogonal to beam direction and iteratively correct roll angle
    # misalignment. Off-centering for roll is towards the right edge of the projection short by
    # twice of sphere diameter.
    # TODO: Understand, if it is necessary to bring the sample back to rotation axis and center
    # axis to projection before calculating roll angle for each iteration. It is possible because
    # in the off-centered state making roll angle adjustments may send the sample outside FOV and
    # then immediate next roll angle estimation will fail.
    _flush_flat_fields()
    await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_tomo)
    logger.debug("Start: off-centering for roll correction.")
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
    logger.debug(f"::roll after iterations: {abs(roll_ang_err)}")
    await align_ctx.move(
        motor=align_ctx.devices.align_motor_obd,
        distance=-off_cent_px_roll * align_ctx.pixel_size_um)
    logger.debug("Done: came back from off-centering for roll correction.")
    logger.debug("Start: centering sample in projection.")
    await center_axis_in_projection(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, logger=logger)
    logger.debug("Done: sample centered in projection.")
    logger.debug("Done: alignment.")
