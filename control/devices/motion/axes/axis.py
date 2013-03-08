'''
Created on Mar 5, 2013

@author: farago
'''
from threading import Thread
from control.devices.device import Device, State
from control.events.dispatcher import dispatcher
from control.devices import device


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


class AxisState(State):
    """Axis status."""
    STANDBY = device.make_state_id()
    MOVING = device.make_state_id()
    POSITION_LIMIT = device.make_state_id()


class ContinuousAxisState(AxisState):
    """Axis status."""
    VELOCITY_LIMIT = device.make_state_id()


class Axis(Device):
    """Base class for everything that moves."""
    def __init__(self, connection, calibration, position_limit=None):
        self._position_limit = position_limit
        self._connection = connection
        self._calibration = calibration
        self._state = None

    def __del__(self):
        self.stop()

    def get_position(self):
        """Get position in set units."""
        return self._calibration.to_user(self._get_position_real())

    def set_position(self, position_user, blocking=False):
        """Set position of the device.

        @param position: position in user units.
        @param blocking: True if the call will block until the movement
                         is done.
        """
        position = self._calibration.to_steps(position_user)

        if self.is_out_of_limits(position_user, self._position_limit):
            limit = Limit(Limit.SOFT)
            self._set_state(AxisState.POSITION_LIMIT)
            raise LimitReached(limit)

        if blocking:
            self._set_position_real(position)
        else:
            t = Thread(target=self._set_position_real, args=(position,))
            t.daemon = True
            t.start()

    def subscribe(self, state, callback):
        dispatcher.subscribe(self, state, callback)

    def wait_for(self, state, callback, timeout=None):
        if self._state == state:
            return callback(self)

        dispatcher.wait(self, state, timeout)
        callback(self)

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

    def _set_position_real(self, position):
        """Call to the device itself for physical position setting.

        This method must be always blocking in order to provide appropriate
        events at appropriate times.

        """
        raise NotImplementedError

    def _get_position_real(self):
        """Call to the device itself for physical position query."""
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

    def get_velocity(self):
        """Get velocity in set units."""
        return self._velocity_calibration.to_user(self._get_velocity_real())

    def set_velocity(self, velocity_user, blocking=False):
        velocity = self._velocity_calibration.to_steps(velocity_user)

        if self.is_out_of_limits(velocity_user, self._velocity_limit):
            limit = Limit(Limit.SOFT)
            self._set_state(ContinuousAxisState.VELOCITY_LIMIT)
            raise LimitReached(limit)

        if blocking:
            self._set_velocity_real(velocity)
        else:
            t = Thread(target=self._set_velocity_real, args=(velocity,))
            t.daemon = True
            t.start()

    def is_hard_velocity_limit_reached(self):
        """Implemented by a particular device."""
        raise NotImplementedError

    def _get_velocity_real(self):
        """Call to the device itself for physical velocity setting."""
        raise NotImplementedError

    def _set_velocity_real(self, velocity):
        """Call to the device itself for physical velocity query.

        This method must be always blocking in order to provide appropriate
        events at appropriate times.

        """
        raise NotImplementedError
