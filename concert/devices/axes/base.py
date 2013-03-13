from threading import Thread
from concert.devices.device import Device, State
from concert.events.dispatcher import dispatcher
from concert.events import type as eventtype


class AxisState(State):
    """Axis status."""
    STANDBY = eventtype.make_event_id()
    MOVING = eventtype.make_event_id()
#    POSITION_LIMIT = eventtype.make_event_id()


class ContinuousAxisState(AxisState):
    """Axis status."""
    VELOCITY_LIMIT = eventtype.make_event_id()


class Axis(Device):
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
        self.set('position', position, blocking)

    def get_position(self):
        """Get the position in user units."""
        return self.get('position')

    def move(self, delta, blocking=False):
        """Move axis by *delta* user units."""
        new_position = self.get_position() + delta
        self.set_position(new_position, blocking)

    def stop(self, blocking=False):
        """Stop the motion."""

        if blocking:
            self._stop_real()
        else:
            t = Thread(target=self._stop_real)
            t.daemon = True
            t.start()

    @property
    def state(self):
        return self._state

    def _set_state(self, state):
        self._state = state
        dispatcher.send(self, state)

    def _stop_real(self):
        """Stop the physical axis.

        This method must be always blocking in order to provide appropriate
        events at appropriate times.

        """
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
    *offset_in_steps* by how many steps the device is away from some zero point.
    """
    def __init__(self, steps_per_unit, offset_in_steps):
        self._steps_per_unit = steps_per_unit
        self._offset = offset_in_steps

    def to_user(self, value_in_steps):
        return value_in_steps / self._steps_per_unit - self._offset

    def to_steps(self, value):
        return (value + self._offset) * self._steps_per_unit


class LimitReached(Exception):
    """Hard limit exception."""
    def __init__(self, limit):
        self._limit = limit

    def __str__(self):
        return repr(self._limit)


class Limit(object):
    """Limit can be soft (set programatically) or hard (determined by a device
    while moving).

    """
    HARD = 0
    SOFT = 1

    def __init__(self, limit_type):
        self._type = limit_type

    @property
    def limit_type(self):
        return self._type

    def __repr__(self):
        if self._type == Limit.HARD:
            limit_type_str = "HARD"
        else:
            limit_type_str = "SOFT"
        return "Limit(type=%s)" % (limit_type_str)

    def __str__(self):
        return repr(self)
