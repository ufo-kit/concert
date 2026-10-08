=========
Processes
=========

Concert provides processes for beamline measurements, including the tomography alignment
procedure described here. Such processes depend on configured devices, calibration data, and,
where applicable, a suitable alignment phantom.

Alignment for Tomography
--------------------------

This procedure centers a sphere alignment phantom on the tomography rotation axis and corrects the
pitch and roll tilt of the rotary stage for parallel-beam CT geometry. The method relies upon
known pixel size for a given magnification and appropriately configured pivot points for the motors,
which it needs to work with.

Geometry and Devices
~~~~~~~~~~~~~~~~~~~~~~

The reference tomography angle is ``offset_tomo``, which defaults to 0 degree. At that angle, the
two sample-alignment translations are parallel and horizontally orthogonal to the beam,
respectively. These motors sit on the tomography motor and rotate with it: their directions relative
to the beam change with tomography angle.

- **tomo_motor** rotates the sample for tomography.
- **align_motor_obd** translates the sample relative to the rotation axis, horizontally orthogonal
  to the beam at the reference angle. The opposing views at 0 and 180 degrees measure this
  centering-error component.
- **align_motor_pbd** translates the sample relative to the rotation axis, parallel to the beam
  at the reference angle. The opposing views at 90 and 270 degrees measure this component.
- **flat_motor** translates the whole rotation stage horizontally, orthogonal to the beam. It
  positions the axis in the projection and moves the sample out of the beam for flat acquisition.
- **z_motor** translates the whole rotation stage vertically.
- **rot_motor_pitch** corrects tilt about an axis orthogonal to the beam. Its error is estimated
  with a deliberate displacement using ``align_motor_pbd``.
- **rot_motor_roll** corrects tilt about an axis parallel to the beam. Its error is estimated
  with a deliberate displacement using ``align_motor_obd``.

Horizontal visibility of a displacement varies continuously with tomography angle; the angle
pairs above isolate the respective components rather than defining ranges of exclusive visibility.

The alignment phantom contains a single absorbing sphere embedded in a rod-like support. A
155--190 micrometer tungsten carbide sphere has been used experimentally; material selection is
application dependent. The sphere must remain identifiable and inside the field of view (FOV)
throughout initialization, trial movements, and deliberate displacement. An initial full rotation
inside the FOV is a prerequisite, but does not guarantee that later adjustments stay inside it.
The surrounding template patch must also fit in the image. Rotation pivots must be configured
so that the prescribed corrections are feasible.

Contexts, Acquisition, and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Device containers and contexts collect the devices and configuration used by the procedure.

.. autoclass:: concert.processes.alignment.AcquisitionDevices
    :members:

.. autoclass:: concert.processes.alignment.AcquisitionContext
    :members:
    :show-inheritance:

.. autoclass:: concert.processes.alignment.AlignmentDevices
    :members:

.. autoclass:: concert.processes.alignment.AlignmentContext
    :members:
    :show-inheritance:

The caller prepares the camera for acquisition, including recording and an appropriate trigger
configuration. The alignment routines call ``camera.grab()``; they do not start recording or
issue software triggers. ``width`` must correspond to the acquired projection width in pixels;
``height`` records the configured height but is not currently used in the correction calculations.
``pixel_size_um`` is the calibrated sample-plane size of the pixel.

When ``flat_field_correct`` is enabled, ``flat_position`` must specify a position that removes the
sample from the beam. The procedure caches one dark frame and one flat frame in its state. It
closes the shutter for the dark frame and opens it for flat acquisition, moving ``flat_motor``
to ``flat_position`` and back. Correction uses :math:`(I-D)/(F-D)` where the flat-minus-dark
value is nonzero, and zero otherwise. Both caches are cleared before the pitch and roll stages,
so those stages acquire fresh references. The current acquisition order grabs the radiograph
before checking the shutter state; callers must account for that ordering, particularly at the
first acquisition. This behavior is retained pending camera investigation.

If ``absorptivity`` is enabled, the routine applies a negative logarithm followed by
``numpy.nan_to_num``. Corrections and conversion precede vertical cropping.
``vert_crop_start`` is inclusive; ``vert_crop_end`` is exclusive. Defaults ``0`` and ``None``
retain the full projection, including its las row. Explicit negative endpoints follow Python
slicing: ``-1`` excludes the last row. Localization coordinates refer to the cropped image.

The caller supplies ``proc_func``, an image-to-binary-mask function that isolates the sphere for
initialization. The identity default is only suitable if the input already serves as the required
mask. Segmentation depends on the imaging conditions; for example::

    import numpy as np
    import skimage.filters as sfl

    def proc_func(projection: np.ndarray) -> np.ndarray:
        smoothed = sfl.gaussian(projection.astype(np.float32), sigma=10)
        return smoothed > sfl.threshold_yen(smoothed)

We decided binary mask generation to be configurable aspect of the routine because the underlying
method can drastically differ based on imaging conditions. One specific method is not guaranteed to
yield the desirable result in all circumstances.

State initialization is a separate call. It records initial motor positions and acquires projections
at 0, 90, 180, and 270 degrees. At each angle it labels the segmentation mask, selects the connected
component with lowest eccentricity, and extracts a patch from the processed projection around its
integer-truncated centroid. The default patch dimension is 200 pixels. The sphere radius is
estimated from the selected component's perimeter divided by :math:`2\pi`, truncated to an integer;
the first nonzero estimate is retained.

Each angle-specific patch is matched against its source projection to obtain a baseline correlation
score. Later localization uses these stored intensity patches, rather than repeating segmentation
or centroid estimation. Template matching is used as localization method. Its correlation maximum
identifies the patch center at integer-pixel coordinates.

Initialization returns the tomography motor to the reference angle.

.. autoclass:: concert.processes.alignment.AlignmentState
    :members:

.. autofunction:: concert.processes.alignment.init_alignment_state

Measurements and Centering
~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`(x_1,y_1)` and :math:`(x_2,y_2)` be the template-localized sphere positions in two
projections 180 degrees apart, with :math:`x` the column and :math:`y` the row coordinate.
The measured quantities, in pixels, are

.. math::

    e_x = \frac{|x_2-x_1|}{2}, \qquad \Delta y = |y_2-y_1|.

Opposing rotation reverses the projected horizontal centering component, giving a separation
of twice that component. In simpler terms if we look at this half circular rotation from the top
:math:`(x_2 - x_1)` is the signed diameter in pixels. We have to halve this value to be used as the
horizontal offset argument with arctangent function. Thus we calculate :math:`e_x` as the horizontal
component, whereas :math:`\Delta y` is the vertical component.

.. autofunction:: concert.processes.alignment.get_sample_shifts

Sample centering uses :math:`e_x` from 0/180-degree views for OBD, then from 90/270-degree views
for PBD. Because the offsets are unsigned, each correction first makes a positive trial movement
``del_dist_lin``, measures the interim error, and reverses the trial movement. If the error
increased, the correction sign is reversed. With pixel size :math:`p`, the sample-centering
correction magnitude is :math:`e_x p / a`, where :math:`a` is ``adjust_move``. We incorporated this
correction adjustment component in the denominator to account for the uncertainties associated with
motor movement. It was observed that approaching the target distance is small iterative moves is a
more reliable strategy with high-precision motors compared to making a big movement. Each component
is corrected until its error is at most ``pixel_err_eps`` or its loop reaches ``max_iterations``.

.. autofunction:: concert.processes.alignment.center_sample_on_axis

With the sphere centered on the axis, its position represents the projected axis at the sphere's
height. At the reference angle, the routine measures absolute vertical and horizontal offsets
from the processed image center :math:`(W/2,H/2)` and adjusts ``z_motor`` followed by ``flat_motor``.
It uses the same trial-and-reversal direction check, but applies the full pixel-to-distance
correction without ``adjust_move``. In this case movement adjustments were not applied because these
motors are reliable for relatively larger movement distances. This positioning supplies the reference
for the deliberate roll displacement; it does not guarantee that subsequent motion stays within the
FOV.

.. autofunction:: concert.processes.alignment.offset_from_projection_center

.. autofunction:: concert.processes.alignment.center_axis_in_projection

Pitch and Roll Correction
~~~~~~~~~~~~~~~~~~~~~~~~~

After external state initialization, the main procedure performs these stages:

1. Preload the six adjustment motors for backlash compensation.
2. Center the sphere on the rotation axis, then center it in the projection.
3. Deliberately displace the sphere parallel to the beam and correct pitch.
4. Return that displacement, displace the sphere orthogonal to the beam, and correct roll.
5. Return that displacement and repeat projection centering.

The deliberate displacement provides a lever arm for observing tilt. In the intended small-tilt
geometry, vertical separation in opposing views is interpreted as twice the lever arm times the
tangent of the tilt. For positive displacement magnitude :math:`r` in pixels, the implemented
angular-error estimate is

.. math::

    \hat{\alpha} = atan2(\Delta_y, 2r).

The result is converted to degrees for motor correction. Pitch uses
:math:`r_{\mathrm{pitch}}=d_{\mathrm{pbd}}/p`, where :math:`d_{\mathrm{pbd}}` is ``off_cent_pbd``.
Roll uses :math:`r_{\mathrm{roll}}=\lfloor W/2\rfloor-4R`, where :math:`R` is the estimated sphere
radius in pixels. This leaves a margin of two sphere diameters from the projection edge when the
axis is centered. The configured geometry must provide a positive displacement and enough FOV
for both stages. Both angular estimates use 0/180-degree views; the deliberate displacement
direction distinguishes pitch from roll. In the arctangent function we use twice horizontal pixel
offset because after deliberate off-centering the distance ``r`` marks radius of a circular motion
and the said circle is directly mapped onto the stage. In this situation from the projections taken
at two terminal angular position the y-component gives the full vertical offset ``Delta_y`` but to
get the full horizontal offset required for arctangent function we need to double the radius and get
the diameter of the circle mapped onto the stage.

Each angular correction makes a positive trial rotation ``del_dist_rot``, measures the interim
error, and reverses the trial. If the error increased, the estimated correction sign is reversed.
The full estimated angle is then applied and measured again. Pitch and roll are handled
sequentially. Attempting a simultaneous adjustment in both will interfere with each other.

The angular stopping threshold is

.. math::

    \alpha_{\mathrm{tol}} = \arctan(s/W),

where :math:`s` is ``pixel_sensitivity``. This is a geometric criterion for vertical pixel
displacement across the image width. Each angular loop stops when its measured error is at most
this threshold or it reaches ``max_iterations``. Every centering or angular loop has its own
iteration cap. Reaching a cap does not raise an exception or prevent subsequent stages.

Stage summaries report residuals with units, tolerances, iteration counts, and convergence status.
Residuals are measured at each stage's completion; later adjustments can affect earlier stages.
They are not a final combined verification, nor experimental evidence of reconstruction resolution.
The procedure returns ``None``; finishing does not establish that every tolerance was met.

.. autofunction:: concert.processes.alignment.align_tomo_stage_parallel_beam

Backlash Compensation
~~~~~~~~~~~~~~~~~~~~~~~

Relative adjustment movements approach their final positions in the positive direction. Defaults
are 0.1 mm for linear motors and 0.1 degrees for rotation motors, assuming the actual backlash is
smaller. Preload moves twice the compensation distance negatively, then positively. Subsequent
positive moves are direct; non-positive moves overshoot negatively by the compensation distance
and return positively by that distance. This assumes that preload establishes the required
mechanical contact and that travel limits permit the overshoot.

.. autoclass:: concert.processes.alignment.BacklashCompRelMovMixin
    :members:

Planned Future Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before computing opposing-view shifts, each projection is matched against its angle-specific
patch. The sample is accepted when the absolute difference between the baseline score and the
absolute current maximum score is below ``score_epsilon``. A failed check raises ``ProcessError``.
This is a correlation-based presence check, not proof that the whole sphere or patch is in view.
Projection-centering measurements do not perform this check.

``checkpoints`` contain initial motor positions only. They are currently not updated or used for
recovery from a failure due reasons such as sample going outside FOV. Planned future work includes
checking camera state before acquiring frame, implementing recovery, validating inputs and template
bounds, providing structured convergence results and implementing a GUI-based workflow.

Usage and Logging
~~~~~~~~~~~~~~~~~~~

The following example assumes all named devices have already been instantiated and configured,
including motor pivots, travel limits, camera exposure, and continuous/internal triggering::

    import logging
    import sys
    from concert.processes.alignment import (
        AcquisitionDevices, AcquisitionContext, AlignmentDevices, AlignmentContext,
        init_alignment_state, align_tomo_stage_parallel_beam,
    )
    from concert.quantities import q

    # Configure once at application startup, before other handlers are installed. This is useful to
    # get some feedback in the console. This will likely change upon implementing a GUI.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    acq_ctx = AcquisitionContext(
        devices=AcquisitionDevices(camera, shutter, tomo_motor, flat_motor, z_motor),
        height=projection_height, width=projection_width,
        flat_field_correct=True, absorptivity=True, flat_position=flat_position,
    )
    align_ctx = AlignmentContext(
        devices=AlignmentDevices(
            rot_motor_pitch, rot_motor_roll, align_motor_pbd, align_motor_obd,
        ),
        pixel_size_um=calibrated_pixel_size_um * q.um,
        proc_func=proc_func,
        viewer=None,
    )

    # For example, execute this coroutine from the configured Concert session.
    async def run_alignment():
        await shutter.open()
        async with camera.recording():
            state = await init_alignment_state(acq_ctx, align_ctx)
            await align_tomo_stage_parallel_beam(acq_ctx, align_ctx, state)

The module uses ordinary Python logging and installs no handlers. ``INFO`` shows stage summaries;
``WARNING`` identifies iteration limits reached above tolerance. Enable ``DEBUG`` on
``concert.processes.alignment`` for detailed measurements, ensuring the handler also accepts that
level. Logger arguments accept a caller-supplied logger.

``basicConfig`` only configures a root logger without existing handlers. In an already configured
Concert session, configure the existing handlers explicitly or supply a dedicated logger with a
``logging.StreamHandler(sys.stdout)``. Configure that logger once and disable propagation if its
own handler would otherwise duplicate root output. Avoid resetting application logging globally.
