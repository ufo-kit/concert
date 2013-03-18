"""
Each axis is associated with a :class:`Calibration` that maps arbitrary
real-world coordinates to devices coordinates. When a calibration is associated
with an axis, the position can be changed with :meth:`Axis.set_position` and
:meth:`Axis.move`::

    from concert.devices.base import LinearCalibration
    from concert.devices.axes.ankatango import ANKATangoDiscreteAxis

    calibration = LinearCalibration(1 / q.mm, 0 * q.mm)
    axis1 = ANKATangoDiscreteAxis(connection, calibration)

    axis.set_position(2 * q.mm, blocking=True)
    axis.move(-0.5 * q.mm)

As long as an axis is moving, :meth:`Axis.stop` will stop the motion.
"""
from concert.base import ConcertObject
from concert.devices.base import State


class Axis(ConcertObject):
    """Base class for everything that moves.

    An axis is used with a *calibration* that conforms to the
    :class:`Calibration` interface to convert between user and device units.

    Exported parameters:
        - ``"position"``: Position of the axis
    """

    def __init__(self, calibration):
        super(Axis, self).__init__()

        self._state = None
        self._register('position',
                       calibration.to_user,
                       calibration.to_steps,
                       None)

    def __del__(self):
        self.stop()

    def set_position(self, position, blocking=False):
        """Set the *position* in user units."""
        return self.set('position', position, blocking)

    def get_position(self):
        """Get the position in user units."""
        return self.get('position')

    def move(self, delta, blocking=False):
        """Move axis by *delta* user units."""
        new_position = self.get_position() + delta
        self.set_position(new_position, blocking)

    def stop(self, blocking=False):
        """Stop the motion."""
        self._launch(self._stop_real, blocking=blocking)

    @property
    def state(self):
        return self._state

    def _set_state(self, state):
        self._state = state
        self.send(self._state)

    def _stop_real(self):
        """Stop the physical axis.

        This method must be always blocking in order to provide appropriate
        events at appropriate times.

        """
        raise NotImplementedError

    def hard_position_limit_reached(self):
        raise NotImplementedError


class ContinuousAxis(Axis):
    """A movable on which one can set velocity.

    This class is inherently capable of discrete movement.

    """
    def __init__(self, position_calibration, velocity_calibration):
        super(ContinuousAxis, self).__init__(position_calibration)
        self._velocity = None
        self._velocity_calibration = velocity_calibration

        self._register('velocity',
                       velocity_calibration.to_user,
                       velocity_calibration.to_steps,
                       None)

    def set_velocity(self, velocity, blocking=False):
        """Set *velocity* of the axis."""
        self.set('velocity', velocity, blocking)

    def get_velocity(self):
        """Get current velocity of the axis."""
        return self.get('velocity')


class AxisState(State):
    """Axis status."""
    STANDBY = "standby"
    MOVING = "moving"


class AxisMessage(object):
    """Axis message."""
    POSITION_LIMIT = "position_limit"
    VELOCITY_LIMIT = "velocity_limit"


class Calibration(object):
    """Interface to convert between user and device units."""

    def to_user(self, value):
        """Return *value* in user units."""
        raise NotImplementedError

    def to_steps(self, value):
        """Return *value* in device units."""
        raise NotImplementedError


class LinearCalibration(Calibration):
    """A linear calibration maps a number of motor steps to a real-world unit.

    *steps_per_unit* tells how many steps correspond to some unit,
    *offset_in_steps* by how many steps the device is away from some zero
    point.
    """
    def __init__(self, steps_per_unit, offset_in_steps):
        super(LinearCalibration, self).__init__()
        self._steps_per_unit = steps_per_unit
        self._offset = offset_in_steps

    def to_user(self, value_in_steps):
        return value_in_steps / self._steps_per_unit - self._offset

    def to_steps(self, value):
        return (value + self._offset) * self._steps_per_unit
