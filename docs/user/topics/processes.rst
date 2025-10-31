=========
Processes
=========

Concert offers implementations for a number of processes, which are essential for running
a measurement at the beamlines. In this page we describe some of these processes along with the
semantics behind them. These implementations offer several configuration options keeping the
inherent uncertainties of working with beamline devices in mind. These processes often rely on
specific calibration artifacts like an alignment phantom. These dependencies of the respective
processes are also described.

Alignment for Tomography
------------------------

Alignment for tomography depends upon several key concepts, which we describe below along with some
pitfalls, which should be considered. Method described below is based on parallel beam CT geometry.
We start by describing relevant devices for the alignment and naming conventions, which are
followed throughout the implementation. Sometimes we used beam direction *bd* as a convention to
be specific and consistent about which direction a motor moves w.r.t to the beam.

- **z_motor**: Moves the rotation stage vertically along orthogonal axis w.r.t beam direction.

- **flat_motor**: Moves the rotation stage horizontally along orthogonal axis w.r.t to beam
  direction. This motor moves the sample away and into the beam, thereby enabling the collection
  of flat fields.

- **tomo_motor**: Seats on top of rotation stage and facilitates tomographic rotation. Axis or
  center of tomographic rotation is the geometric center of this motor.

- **align_motor_obd**: Seats on top of `tomo_motor` and moves the sample *horizontally* towards
  or away from center of rotation along axis, which is *orthogonal to beam direction (obd)*.
  Conventionally, we associate the rotational span of (0 - 180) degrees with this motor, means if
  we off-center our sample with this motor, then the horizontal offset from center would be visible
  in projection only for the angular range of (0 - 180) degrees of `tomo_motor`.

- **align_motor_pbd**: Seats on top of `tomo_motor` and moves the sample *horizontally* towards
  or away from center of rotation along axis, which is *parallel to beam direction (pbd)*.
  Conventionally, we associate the rotational span of (90 - 270) degrees with this motor, means if
  we off-center our sample with this motor, then the horizontal offset from center would be visible
  in projection only for the angular range of (90 - 270) degrees of `tomo_motor`.

- **rot_motor_pitch**: Rotates the stage to account for pitch angle misalignment. Axis around
  which this motor rotates is spanned orthogonal to the beam direction. We off-center our sample
  using *align_motor_pbd* to estimate the angular error which is relevant for this motor.

- **rot_motor_roll**: Rotates the stage to account for roll angle misalignment. Axis around
  which this motor rotates is spanned parallel to the beam direction. We off-center our sample
  using *align_motor_obd* to estimate the angular error which is relevant for this motor.

Our sample for tomographic alignment is a single metal sphere of high-absorbing compound embedded
in a rod-like structure. As an example, we used 155-190 microns Tungsten Carbide sphere for our
phantom. Choice of material in this regard is experimental.

Misalignment occurs from a very small tilt of the rotation stage either in the direction orthogonal
to the beam (roll error) and/or parallel direction to the beam (pitch error). Before we move on to
the details of calculating this tilt we'd take the following aspects of the linear alignment motors
into account.

Both **align_motor_obd** and **align_motor_pbd** take the sample towards or away from the center of
rotation. It is useful to think about the top-view perspective of the rotation stage in this regard.
When they take the sample away from center of rotation the resulting offset from center becomes the
radius of the rotation w.r.t each direction. We can measure these distances from two projections
taken with an angular offset of 180 degrees and it is useful to bring the sample back to its center
of rotation. To measure these distances we need to keep in mind, that  alignment motors are placed
on top of rotation motor. It means, although the displacements of these motors are linear, the
axis along which their respective displacements take place, may rotate. That's why we refrain from
strictly associating cartesian-x or cartesian-y axes to either of the motors and tried to generalize
their displacements w.r.t to beam direction. Let's assume, that the angular range of [0, 180]
degrees is aligned with motor **align_motor_obd**, which moves the sample horizontally on top of
rotation plane in orthogonal direction w.r.t beam. While rotation stage is aligned any explicit
displacements made using the other alignment motor **align_motor_pbd** will not be perceived for
this angular range. For that we need to rotate the stage by 90 degrees first and then we can measure
the displacement made by motor **align_motor_pbd** from two projections taken across [90, 270]
degrees. Before starting alignment we assume that all the aforementioned motors are at arbitrary
positions, but we need to ensure, **that the sample can be rotated 360 degrees inside FOV and pivot
points of rotation motors are set appropriately**.

Backlash Compensation
~~~~~~~~~~~~~~~~~~~~~
Backlash refers to the small amount of lost motion or mechanical slack that occurs whenever a
motion system reverses direction. In high-precision linear or rotary stages, it arises from
clearances between mating components such as screw threads, gears, or couplings. When direction
changes, the driving element must take up this slack before the driven part begins to move,
resulting in a temporary position error. It can prevent us from converging to minima of the
angular errors and should be taken into account for motor movement.

To compensate for backlash we take a uni-directional movement approach with an assumption, that
any +ve directional movement of the motor is feasible without the mechanical slack. When we change
direction from +ve to -ve or vice versa we need to take backlash into account. Our objective is
to make relative movements in a way that backlash remains constant and consistently in the -ve
direction when we make the final move which always has to be in the +ve direction according to
our assumption. Concretely, we overshoot towards -ve direction by a small distance in relative
sense and then come back by the same amount toward the +ve direction. These small relative
distances should be big-enough to cover the systematic backlash. Accuracy in motor movement in
this manner relies on the **preload** step at the beginning, where we try to ensure that gears
are touching the face in the +ve direction when we initiate any relative movement.

.. autoclass:: concert.processes.alignment.BacklashCompRelMovMixin
    :members:

Context and State
~~~~~~~~~~~~~~~~~

Contexts are basically dataclasses used for encapsulating all the devices used for alignment and
their respective configurations for the algorithm. They help in organizing the implementation
and facilitates ease of use.

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

State encapsulates the essence of anomaly detection during the execution of the algorithm. Most
common anomaly is the sample going outside FOV during the alignment, which is possible for various
reasons. We want the algorithm to detect and gracefully react to such circumstances by either
attempting a recovery or at the very least leaving the system in a stable state. Class
:class:`concert.processes.alignment.AlignmentState` provides the building blocks for the same.

.. autoclass:: concert.processes.alignment.AlignmentState
    :members:

Centering Sample on Rotation Axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We measure the horizontal offset between the sample and axis of rotations using two projections
acquired 180 degrees apart and adjust the motors to move the sample toward the axis. The offsets
computed for **align_motor_obd** and **align_motor_pbd** are independent of each other and need to
be handled accordingly. Upon centering the sample on rotation axis we should not perceive any
horizontal movement of sample upon rotation.

.. autofunction:: concert.processes.alignment.center_sample_on_axis

Centering Rotation Axis on Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Centering the rotation axis at the middle of the projection could be helpful in several occasions.
One specific example is, when we off-center the sample from rotation axis to calculate the roll
error, where centering the axis on projection ensures that the off-centering distance for the motor
**align_motor_obd** is derived correctly. This routine is generally helpful in preventing the sample
going outside FOV during alignment but cannot guarantee it.

.. autofunction:: concert.processes.alignment.center_axis_in_projection

Off-centering Sample from Axis and Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting the alignment all the aforementioned motors are at arbitrary positions but we ensure
that the sample can be rotated 360 degrees inside FOV. A preload for backlash compensation and
initialization of state for anomaly detection and optional recovery is performed, which are
described above. We start by first centering the sample on rotation axis and centering the rotation
axis in the middle of the projection. This two steps recover the system from the initial arbitrary
state to a stable state from which we can predictably move the motors.

We then deliberately off-center the alignment motors **align_motor_obd** and **align_motor_pbd**
and estimate the roll and pitch errors respectively. In presence of misalignment, we compute the
vertical offsets of the sample using two projections acquired 180 degrees apart, which helps us to
estimate the angular errors in each case. We refrain from handling two motors simultaneously,
which can lead to ambiguity in the observed vertical offsets. In both cases respective rotation
motors **rot_motor_roll** and **rot_motor_pitch** are iteratively adjusted with the estimated
errors, which advances the system towards the aligned state.

.. autofunction:: concert.processes.alignment.align_tomography_generic





    



