'''
Created on Mar 5, 2013

@author: farago
'''
from threading import Thread
from control.devices.device import Device, State
from control.events.dispatcher import dispatcher
from control.events import type as eventtype


class AxisState(State):
    """Axis status."""
    STANDBY = eventtype.make_event_id()
    MOVING = eventtype.make_event_id()
    POSITION_LIMIT = eventtype.make_event_id()


class ContinuousAxisState(AxisState):
    """Axis status."""
    VELOCITY_LIMIT = eventtype.make_event_id()


class Axis(Device):
    """Base class for everything that moves."""
    def __init__(self, connection, calibration, position_limit=None):
        super(Axis, self).__init__()

        self._position_limit = position_limit
        self._connection = connection
        self._state = None

        self._register('position',
                       calibration.to_user,
                       calibration.to_steps,
                       None)

    def __del__(self):
        self.stop()

    def set_position(self, position, blocking=False):
        self.set('position', position, blocking)

    def get_position(self):
        return self.get('position')

    def stop(self, blocking=False):
        """Stop the motion."""

        if blocking:
            self._stop_real()
        else:
            t = Thread(target=self._stop_real)
            t.daemon = True
            t.start()

    def is_out_of_limits(self, value, limit):
        """Check if we are outside of the soft limits."""
        if limit is None:
            return False
        return limit and value < limit[0] or value > limit[1]

    def is_hard_position_limit_reached(self):
        """Implemented by a particular device."""
        raise NotImplementedError

    @property
    def position_limit(self):
        return self._position_limit

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
    def __init__(self, connection, position_calibration, velocity_calibration,
                 position_limit=None, velocity_limit=None):

        super(ContinuousAxis, self).__init__(connection, position_calibration,
                                             position_limit)
        self._velocity_limit = velocity_limit
        self._velocity = None
        self._velocity_calibration = velocity_calibration

        self._register('velocity',
                       velocity_calibration.to_user,
                       velocity_calibration.to_steps,
                       None)

    def set_velocity(self, velocity, blocking=False):
        self.set('velocity', velocity, blocking)

    def get_velocity(self):
        return self.get('velocity')

    def is_hard_velocity_limit_reached(self):
        """Implemented by a particular device."""
        raise NotImplementedError


class _Calibration(object):
    def to_user(self, value):
        raise NotImplementedError

    def to_steps(self, value):
        raise NotImplementedError


class LinearCalibration(_Calibration):
    """Represents a linear calibration.

    A linear calibration maps a number of motor steps to a real-world unit
    system.

    """
    def __init__(self, steps_per_unit, offset_in_steps):
        self._steps_per_unit = steps_per_unit
        self._offset = offset_in_steps

    def to_user(self, value_in_steps):
        """Convert value_in_steps to user units"""
        return value_in_steps / self._steps_per_unit + self._offset

    def to_steps(self, value):
        """Convert user unit value to motor steps"""
        return (value - self._offset) * self._steps_per_unit


class LimitReached(Exception):
    """Any limit (hard or soft) exception."""
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
