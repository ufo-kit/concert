"""
The :mod:`.tomoalignment` module aligns rotation axis for tomography
experiments. It uses :mod:`.rotationaxis` module as a measure of rotation
axis inclination towards :math:`y`-axis.

**Algorithm**

The procedure consists of the following steps:

#. Acquire a set of images with rotated sample
#. Determine the axis inclination angles.
#. Rotate the sample according to the measured angles.

The mentioned steps are repeated until a certain minimum threshold is reached.
"""
import numpy as np
from concert.quantities import q
from concert.asynchronous import dispatcher, wait, async
from concert.processes.base import Process


class Aligner(Process):

    """Tomographic rotation axis alignment procedure."""
    # Aligned message
    AXIS_ALIGNED = "axis-aligned"

    def __init__(self, axis_measure, scanner, x_motor, z_motor=None):
        """Contructor. *axis_measure* provides axis of rotation angular
        misalignment data, *scanner* provides image sequences with
        sample rotated around axis of rotation.
        *x_motor* turns the sample around x-axis, *z_motor* is optional
        and turns the sample around z-axis
        """
        super(Aligner, self).__init__(None)
        self._axis_measure = axis_measure
        self._scanner = scanner
        self.x_motor = x_motor
        self.z_motor = z_motor

    @async
    def run(self, absolute_eps=0.1 * q.deg):
        """
        run(absolute_eps=0.1*q.deg)

        The procedure finishes when it finds the minimum angle between an
        ellipse extracted from the sample movement and respective axes or the
        found angle drops below *absolute_eps*. The axis of rotation after
        the procedure is (0,1,0), which is the direction perpendicular
        to the beam direction and the lateral direction.
        """
        # Sometimes both z-directions need to be tried out because of the
        # projection ambiguity.
        z_direction = -1

        x_last = None
        z_last = None
        z_turn_counter = 0

        while True:
            self._axis_measure.images = self._scanner.run().result()[1]
            x_angle, z_angle = self._axis_measure()

            x_better = True if self.z_motor is not None and\
                (x_last is None or np.abs(x_angle) < x_last) else False
            z_better = True if z_last is None or np.abs(z_angle) < z_last\
                else False

            if z_better:
                z_turn_counter = 0
            elif z_turn_counter < 1:
                # We might have rotated in the opposite direction because
                # of the projection ambiguity. However, that must be picked up
                # in the next iteration, so if the two consequent angles
                # are worse than the minimum, we have the result.
                z_better = True
                z_direction = -z_direction
                z_turn_counter += 1

            x_future, z_future = None, None
            if z_better and np.abs(z_angle) >= absolute_eps:
                x_future = self.x_motor.move(z_direction * z_angle)
            if x_better and np.abs(x_angle) >= absolute_eps:
                z_future = self.z_motor.move(x_angle)
            elif (np.abs(z_angle) < absolute_eps or not z_better):
                # The newly calculated angles are worse than the previous
                # ones or the absolute threshold has been reached,
                # stop iteration.
                dispatcher.send(self, self.AXIS_ALIGNED)
                break

            wait([future for future in [x_future, z_future]
                  if future is not None])

            x_last = np.abs(x_angle)
            z_last = np.abs(z_angle)
