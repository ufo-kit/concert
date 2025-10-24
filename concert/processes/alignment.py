from dataclasses import dataclass
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
import skimage.filters as sfl
import skimage.feature as sft
import skimage.measure as sms
from skimage.measure._regionprops import RegionProperties
import skimage.registration as skr
from concert.coroutines.base import background
from concert.imageprocessing import flat_correct
from concert.devices.motors.base import LinearMotor, RotationMotor
from concert.ext.viewers import PyQtGraphViewer
from concert.quantities import Quantity, q
from concert.typing import ArrayLike, Motor_T
from concert.processes.common import AcquisitionContext, AlignmentContext
from concert.processes.common import get_noop_logger, ProcessError


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
    dark: Optional[ArrayLike] = None
    flat: Optional[ArrayLike] = None
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
        return val

    def sample_in_FOV(self, frame: ArrayLike, angle: int) -> bool:
        """
        Evaluates if sample is inside FOV for the given `frame` and `angle`.

        We derive a confidence score by matching a template patch for the given `angle` to the
        `frame`. This score is then used to compute a relative certainty against respective baseline
        score for the `angle`. Baseline score is obtained by matching the template patch with the
        projection it is expected form, hence baseline scores are approximately 1, representing a
        definite match. While evaluating if the sample is inside FOV we take the same template patch
        for the appropriate angle and get a new `score`. If template is indeed matched (sample in
        FOV) then new `score` is approximately equal to `baseline score` making `relative
        likelihood` high and in turn `uncertainty` low. In contrast, when new `score` is low (sample
        is likely outside FOV) then `relative likelihood` is low making `uncertainty` high. We
        threshold on this uncertainty. In principle, we are asking, how uncertain we are that the
        sample is inside FOV against the maximum threshold `uncertainty` that we want to allow.

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


async def acquire_single(ctx: AcquisitionContext, align_state: AlignmentState) -> ArrayLike:
    """
    Acquires a single frame using context provided for acquisition.

    :param ctx: context for acquisition
    :type ctx: `concert.processes.common.AcquisitionContext`
    :param align_state: state for alignment
    :type align_state: `concert.processes.common.AcquisitionState`
    :return: acquired frame
    :rtype: `concert.typing.ArrayLike`
    """
    # TODO: Why we need to do the following camera-specific stuff below ?
    # if await ctx.devices.camera.get_state() == 'recording':
    #     await ctx.devices.camera.stop_recording()
    # await ctx.devices.camera['trigger_source'].stash()
    # await ctx.devices.camera.set_trigger_source(ctx.devices.camera.trigger_sources.SOFTWARE)
    # try:
    frame: ArrayLike = await ctx.devices.camera.grab()
    try:
        if ctx.flat_field_correct:
            if align_state.dark is None:
                if await ctx.devices.shutter.get_state() != 'closed':
                    await ctx.devices.shutter.close()
                # await ctx.devices.camera.trigger()
                align_state.dark = await ctx.devices.camera.grab()
            if await ctx.devices.shutter.get_state() != 'open':
                await ctx.devices.shutter.open()
            if align_state.flat is None:
                radio_pose: Quantity = await ctx.devices.flat_motor.get_position()
                await ctx.devices.flat_motor.set_position(ctx.flat_position)
                # await ctx.devices.camera.trigger()
                align_state.flat = await ctx.devices.camera.grab()
                await ctx.devices.flat_motor.set_position(radio_pose)
            frame = flat_correct(radio=frame, flat=align_state.flat, dark=align_state.dark)
        if ctx.absorptivity:
            frame = np.nan_to_num(-np.log(frame))
    finally:
        pass
        # await ctx.devices.shutter.close()
    return np.asarray(frame)


async def init_alignment_state(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        logger: logging.Logger = get_noop_logger()) -> AlignmentState:
    """
    Initializes alignment state.

    - STEP 1: Record relevant motor positions before starting with the alignment
    - STEP 2: For each terminal angle, grab a frame, extract a patch containing the sample, use
    template matching to get a baseline certainty score for sample definitely being inside FOV.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :return: initial alignment state
    :rtype: `concert.processes.common.AlignmentState`
    """
    logger.debug=print # TODO: For quick debugging, remove later

    logger.debug("Start: state initialization before alignment.")
    state = AlignmentState(checkpoints={}, patches={}, baseline_scores={})
    # Record all relevant motor positions
    for motor_str in ["flat_motor", "z_motor"]:
        state.checkpoints[motor_str] = await getattr(acq_ctx.devices, motor_str).get_position()
    for motor_str in ["align_motor_obd", "align_motor_pbd", "rot_motor_roll", "rot_motor_pitch"]:
        state.checkpoints[motor_str] = await getattr(align_ctx.devices, motor_str).get_position()
    # Record patches and baseline scores for the terminal angles
    for angle in [0, 90, 180, 270]:
        await acq_ctx.devices.tomo_motor.set_position(angle * q.deg + align_ctx.offset_rot_tomo)
        frame: ArrayLike = await acquire_single(ctx=acq_ctx, align_state=state)
        mask: ArrayLike = align_ctx.proc_func(frame)
        regions: List[RegionProperties] = sms.regionprops(label_image=sms.label(mask))
        cnt_y, cnt_x = sorted(regions, key=lambda r: r.eccentricity)[0].centroid
        dim = state.dim // 2
        state.patches[str(angle)] = frame[
            int(cnt_y) - dim:int(cnt_y) + dim, int(cnt_x)- dim:int(cnt_x) + dim]
        state.baseline_scores[str(angle)] = np.max(
            sft.match_template(image=frame, template=state.patches[str(angle)], pad_input=True))
        viewer = await PyQtGraphViewer(show_refresh_rate=True)
        await viewer.set_title(f"Angle = {str(angle)}")
        await viewer.show(state.patches[str(angle)])
    await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_rot_tomo)
    logger.debug(f"Done: state initialized as:\n{state}")
    return state


async def get_sample_shifts(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        tomo_angle: Quantity) -> Tuple[float, float]:
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
    def _clean_mask(frame: ArrayLike) -> ArrayLike:
        """Creates a clean mask keeping sphere region"""
        regions = sms.regionprops(label_image=sms.label(frame))
        region = sorted(regions, key=lambda r: r.eccentricity)[0]
        mask = np.zeros_like(frame, dtype=np.float32)
        mask[tuple(zip(*region.coords))] = 1.0
        return mask

    await acq_ctx.devices.tomo_motor.set_position(tomo_angle + align_ctx.offset_rot_tomo)
    ref_frame: ArrayLike = await acquire_single(ctx=acq_ctx, align_state=align_state)
    if not align_state.sample_in_FOV(frame=ref_frame, angle=tomo_angle.magnitude):
        raise ProcessError("sample went outside FOV before rotation, aborting")
    ref_img: ArrayLike = _clean_mask(frame=align_ctx.proc_func(ref_frame))
    if align_ctx.viewer: await align_ctx.viewer.show(ref_img)
    await acq_ctx.devices.tomo_motor.move(180 * q.deg)
    mov_frame: ArrayLike = await acquire_single(ctx=acq_ctx, align_state=align_state)
    if not align_state.sample_in_FOV(frame=mov_frame, angle=tomo_angle.magnitude + 180):
        raise ProcessError("sample went outside FOV after rotation, aborting")
    mov_img: ArrayLike = _clean_mask(frame=align_ctx.proc_func(mov_frame))
    if align_ctx.viewer: await align_ctx.viewer.show(mov_img)
    shift_yx, _, _ = skr.phase_cross_correlation(
        reference_image=ref_img, moving_image=mov_img, upsample_factor=4)
    await acq_ctx.devices.tomo_motor.move(-180 * q.deg)
    return shift_yx[0], abs(shift_yx[1]) / 2


async def move_relative(
        motor: Motor_T,
        move_by: Quantity,
        align_ctx: AlignmentContext,
        logger: logging.Logger = get_noop_logger()) -> None:
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
    Accuracy in motor movement in this manner relies on the preload step at the beginning, where we
    try to ensure that gears are touching the face in the +ve direction while backlash remains in
    the -ve direction.

    :param motor: linear or rotation motor used for alignment
    :type motor: `concert.typing.Motor_T`
    :param move_by: relative distance to move the motor
    :type move_by: `concert.quantities.Quantity`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later
    if np.sign(move_by) > 0:
        await motor.move(move_by)
        return
    bl_comp: Quantity = align_ctx.bl_comp_rel_rot if isinstance(motor, RotationMotor) \
        else align_ctx.bl_comp_rel_lin
    await motor.move(move_by - bl_comp)
    await motor.move(bl_comp)


async def center_sample_on_axis(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
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

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param align_state: alignment state
    :type align_state: `concert.processes.common.AlignmentState`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later

    async def _move_corrective(motor: LinearMotor, offset: float, tomo_angle: Quantity) -> float:
        await move_relative(
            motor=motor, move_by=align_ctx.delta_move_mm, align_ctx=align_ctx, logger=logger)
        _, _offset = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state,
            tomo_angle=tomo_angle)
        await move_relative(
            motor=motor, move_by=-align_ctx.delta_move_mm, align_ctx=align_ctx, logger=logger)
        if _offset > offset:
            offset = -offset
        await move_relative(
            motor=motor, move_by=offset * align_ctx.pixel_size_um, align_ctx=align_ctx,
            logger=logger)
        _, _new_offset = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state,
            tomo_angle=tomo_angle)
        return _new_offset
    
    # Get initial distances from center of rotation
    _, offset_obd = await get_sample_shifts(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=0 * q.deg)
    _, offset_pbd = await get_sample_shifts(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=90 * q.deg)
    logger.debug(f">> before: offset_obd = {offset_obd} offset_pbd = {offset_pbd}")
    while offset_obd > align_ctx.pixel_err_eps or offset_pbd > align_ctx.pixel_err_eps:
        # Make step adjustment for alignment motor orthogonal to beam direction
        if offset_obd > align_ctx.pixel_err_eps:
            offset_obd = await _move_corrective(
                motor=align_ctx.devices.align_motor_obd, offset=offset_obd, tomo_angle=0 * q.deg)
        # Make step adjustment for alignment motor parallel to beam direction.
        if offset_pbd > align_ctx.pixel_err_eps:
            offset_pbd = await _move_corrective(
                motor=align_ctx.devices.align_motor_pbd, offset=offset_pbd, tomo_angle=90 * q.deg)
    logger.debug(f">> after: offset_obd = {offset_obd} offset_pbd = {offset_pbd}")


async def offset_from_projection_center(
        acq_ctx: AcquisitionContext,
        align_state: AlignmentState,
        tomo_angle: Quantity) -> Tuple[float, float]:
    """
    Derives the distance in pixels between geometric center of the projection and center of mass
    in XY detector plane. These values are useful to adjust tomographic stage motor (horizontally
    orthogonal to beam direction) and z-motor (vertically orthogonal to beam direction) to center a
    sample in the projection.

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_state: managed state for alignment
    :type align_state: `concert.processes.common.AlignmentContext`
    :param tomo_angle: angle of the tomo rotation motor
    :type tomo_angle: `concert.quantities.Quantity`
    :return: distance between the geometric center of projection and center of mass 
    :rtype: Tuple[float, float]
    """
    frame: ArrayLike = await acquire_single(ctx=acq_ctx, align_state=align_state)
    patch = align_state.patches[str(tomo_angle.magnitude)]
    matched = sft.match_template(image=frame, template=patch, pad_input=True)
    indices = np.unravel_index(np.argmax(matched), matched.shape)
    xcm, ycm = indices[::-1]
    yc_proj, xc_proj = frame.shape[0] / 2, frame.shape[1] / 2
    z_offset, stage_offset = abs(
        yc_proj - ycm), abs(xc_proj - xcm)
    return z_offset, stage_offset


async def center_stage_in_projection(
        acq_ctx: AcquisitionContext,
        align_ctx: AlignmentContext,
        align_state: AlignmentState,
        tomo_angle: Quantity,
        logger: logging.Logger = get_noop_logger()) -> None:
    """
    Adjusts the vertical motor(`z_motor`) and horizontal motor(`flat_motor`) to put the center of
    mass of the sample in the middle of the projection.
   
    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param align_state: managed state for alignment
    :type align_state: `concert.processes.common.AlignmentState`
    :param tomo_angle: angle of the tomo rotation motor
    :type tomo_angle: `concert.quantities.Quantity`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later
    z_offset, stage_offset = await offset_from_projection_center(
        acq_ctx=acq_ctx, align_state=align_state, tomo_angle=tomo_angle)
    logger.debug(f">> before: z_offset = {z_offset} stage_offset = {stage_offset}")
    while z_offset > align_ctx.pixel_err_eps or stage_offset > align_ctx.pixel_err_eps:
        # Make step adjustment for z-motor (vertically orthogonal to beam direction)
        if z_offset > align_ctx.pixel_err_eps:
            await move_relative(
                motor=acq_ctx.devices.z_motor, move_by=align_ctx.delta_move_mm,
                align_ctx=align_ctx, logger=logger)
            _z_offset, _ = await offset_from_projection_center(
                acq_ctx=acq_ctx, align_state=align_state, tomo_angle=tomo_angle)
            await move_relative(
                motor=acq_ctx.devices.z_motor, move_by=-align_ctx.delta_move_mm,
                align_ctx=align_ctx, logger=logger)
            if _z_offset > z_offset:
                z_offset = -z_offset
            await move_relative(
                motor=acq_ctx.devices.z_motor, move_by=z_offset * align_ctx.pixel_size_um,
                align_ctx=align_ctx, logger=logger)
        # Make step adjustment for stage motor (horizontally orthogonal to beam direction)
        if stage_offset > align_ctx.pixel_err_eps:
            await move_relative(
                motor=acq_ctx.devices.flat_motor, move_by=align_ctx.delta_move_mm,
                align_ctx=align_ctx, logger=logger)
            _, _stage_offset = await offset_from_projection_center(
                acq_ctx=acq_ctx, align_state=align_state, tomo_angle=tomo_angle)
            await move_relative(
                motor=acq_ctx.devices.flat_motor, move_by=-align_ctx.delta_move_mm,
                align_ctx=align_ctx, logger=logger)
            if _stage_offset > stage_offset:
                stage_offset = -stage_offset
            await move_relative(
                motor=acq_ctx.devices.flat_motor, move_by=stage_offset * align_ctx.pixel_size_um,
                align_ctx=align_ctx, logger=logger)
        z_offset, stage_offset = await offset_from_projection_center(
            acq_ctx=acq_ctx, align_state=align_state, tomo_angle=tomo_angle)
    logger.debug(f">> after: z_offset = {z_offset} stage_offset = {stage_offset}")


@background
async def align_rotation_stage_comparative(
    acq_ctx: AcquisitionContext,
    align_ctx: AlignmentContext,
    align_state: AlignmentState,
    logger: logging.Logger = get_noop_logger()) -> None:
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
    degrees of rotation, generally we want to first center the sample on rotation axis preparatory
    step before off-centering for alignment. It provides us with a clean slate to start actual
    steps of alignment.
    
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

    :param acq_ctx: context for acquisition
    :type acq_ctx: `concert.processes.common.AcquisitionContext`
    :param align_ctx: context for alignment
    :type align_ctx: `concert.processes.common.AlignmentContext`
    :param align_state: state managed for alignment
    :type align_state: `concert.processes.common.AlignmentState`
    :param logger: optional logger
    :type logger: logging.Logger
    """
    logger.debug=print # TODO: For quick debugging, remove later

    async def _preload(motor: Motor_T) -> None:
        """
        Ensures that gear of the `motor` touches the face on the +ve side to ensure accuracy of
        subsequent moves with backlash compensation.
        """
        bl_comp: Quantity = align_ctx.bl_comp_rel_rot if isinstance(motor, RotationMotor) \
        else align_ctx.bl_comp_rel_lin
        await motor.move(-2 * bl_comp)
        await motor.move(2 * bl_comp)        

    async def _get_roll() -> Quantity:
        """Estimates roll angle error"""
        await move_relative(
            motor=align_ctx.devices.align_motor_obd, move_by=align_ctx.offset_lin_obd,
            align_ctx=align_ctx, logger=logger)
        ver_obd_px, hor_obd_px = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=0 * q.deg)
        await move_relative(
            motor=align_ctx.devices.align_motor_obd, move_by=-align_ctx.offset_lin_obd,
            align_ctx=align_ctx, logger=logger)
        return np.rad2deg(np.arctan2(ver_obd_px, 2 * hor_obd_px)) * q.deg
    
    async def _step_roll(curr_roll: Quantity) -> Quantity:
        """Makes a corrective step for roll angle error"""
        delta_move: Quantity = np.sign(curr_roll) * align_ctx.delta_move_deg
        await move_relative(
            motor=align_ctx.devices.rot_motor_roll, move_by=delta_move, align_ctx=align_ctx,
            logger=logger)
        _roll: Quantity = await _get_roll()
        await move_relative(
            motor=align_ctx.devices.rot_motor_roll, move_by=-delta_move, align_ctx=align_ctx,
            logger=logger)
        if abs(_roll) > abs(curr_roll):
            curr_roll = -curr_roll
        await move_relative(
            motor=align_ctx.devices.rot_motor_roll, move_by=curr_roll, align_ctx=align_ctx,
            logger=logger)
        return await _get_roll()

    async def _get_pitch() -> Quantity:
        """Estimates pitch angle error"""
        await move_relative(
            motor=align_ctx.devices.align_motor_pbd, move_by=align_ctx.offset_lin_pbd,
            align_ctx=align_ctx, logger=logger)
        ver_pbd_px, _ = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=0 * q.deg)
        _, hor_pbd_px = await get_sample_shifts(
            acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, tomo_angle=90 * q.deg)
        await move_relative(
            motor=align_ctx.devices.align_motor_pbd, move_by=-align_ctx.offset_lin_pbd,
            align_ctx=align_ctx, logger=logger)
        return np.rad2deg(np.arctan2(ver_pbd_px, 2 * hor_pbd_px)) * q.deg
    
    async def _step_pitch(curr_pitch: Quantity) -> Quantity:
        """Makes a corrective step for pitch angle error"""
        delta_move: Quantity = np.sign(curr_pitch) * align_ctx.delta_move_deg
        await move_relative(
            motor=align_ctx.devices.rot_motor_pitch, move_by=delta_move,
            align_ctx=align_ctx, logger=logger)
        _pitch: Quantity = await _get_pitch()
        await move_relative(
            motor=align_ctx.devices.rot_motor_pitch, move_by=-delta_move,
            align_ctx=align_ctx, logger=logger)
        if abs(_pitch) > abs(curr_pitch):
            curr_pitch = -curr_pitch
        await move_relative(
            motor=align_ctx.devices.rot_motor_pitch, move_by=curr_pitch,
            align_ctx=align_ctx, logger=logger)
        return await _get_pitch()
    
    def _flush_flat_fields() -> None:
        """Force acquiring new flat fields"""
        align_state.dark = None
        align_state.flat = None

    func_name = "align_rotation_stage_comparative"
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    logger.debug(f"{func_name}")
    logger.debug("#" * 3 * len(f"Start: {func_name}"))
    # STEP 0: Sample is brought into FOV and a 360 degrees rotation is possible without sending it
    # outside FOV.
    # STEP 1: Preload and initialize alignment state.
    logger.debug("Start: preload for backlash compensation.")
    await _preload(acq_ctx.devices.flat_motor)
    await _preload(acq_ctx.devices.z_motor)
    await _preload(align_ctx.devices.align_motor_obd)
    await _preload(align_ctx.devices.align_motor_pbd)
    await _preload(align_ctx.devices.rot_motor_roll)
    await _preload(align_ctx.devices.rot_motor_pitch)
    logger.debug("Done: preload.")
    # STEP 2: Center sample into projection in two steps, firstly centering the sample to the axis
    # and then centering the tomo stage to the projection by center of mass.
    logger.debug("Start: centering sample on axis.")
    await center_sample_on_axis(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state, logger=logger)
    logger.debug("Done: sample centered on axis.")
    # NOTE: Alignment metric is a measure of resolution sensitivity. We derive it as an angular
    # error threshold which represents maximum `align_ctx.pixel_sensitivity` pixels of resolution
    # loss vertically against `acq_ctx.width` pixels horizontally.
    align_metric: Quantity = np.rad2deg(
        np.arctan(align_ctx.pixel_sensitivity / acq_ctx.width)) * q.deg
    logger.debug(f"Calculated: alignment metric: {align_metric}")
    # STEP 3: Off-center the sample orthogonal to beam direction and iteratively correct roll angle
    # misalignment.
    _flush_flat_fields()
    logger.debug("Start: alignment.")
    roll_iter = 0
    roll_ang_err: Quantity = await _get_roll()
    logger.debug(f"::roll before iterations: {abs(roll_ang_err)}")
    while abs(roll_ang_err) > align_metric:
        roll_ang_err = await _step_roll(curr_roll=roll_ang_err)
        roll_iter += 1
        logger.debug(f"::::roll-correction iteration: {roll_iter} roll: {abs(roll_ang_err)}")
        if roll_iter == align_ctx.max_iterations:
            logger.debug("::::max roll iterations reached")
    logger.debug(f"::roll after iterations: {abs(roll_ang_err)}")
    # STEP 4: Off-center the sample parallel to beam direction and iteratively correct pitch
    # (lamino) angle misalignment.
    _flush_flat_fields()
    pitch_iter = 0
    pitch_ang_err: Quantity = await _get_pitch()
    logger.debug(f"::pitch before iterations: {abs(pitch_ang_err)}")
    while abs(pitch_ang_err) > align_metric:
        pitch_ang_err = await _step_pitch(curr_pitch=pitch_ang_err)
        pitch_iter += 1
        logger.debug(f"::::pitch-correction iteration: {pitch_iter} pitch: {abs(pitch_ang_err)}")
        if pitch_iter == align_ctx.max_iterations:
            logger.debug("::::max pitch iterations reached")
    logger.debug(f"::pitch after iterations: {abs(pitch_ang_err)}")
    await acq_ctx.devices.tomo_motor.set_position(0 * q.deg + align_ctx.offset_rot_tomo)
    logger.debug("Start: centering sample in projection.")
    await center_stage_in_projection(
        acq_ctx=acq_ctx, align_ctx=align_ctx, align_state=align_state,
        tomo_angle=0 * q.deg, logger=logger)
    logger.debug("Done: sample centered in projection.")
    logger.debug("Done: alignment.")