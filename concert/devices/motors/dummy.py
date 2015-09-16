"""Motor Dummy."""
import random
from concert.base import HardLimitError, Quantity
from concert.quantities import q
from concert.devices.motors import base
from time import sleep
import numpy as np

MOVEMENT_TIME_STEPS = 0.01*q.s


class _PositionMixin(object):
    def __init__(self):
        self._position = 0 * q.mm
        self._moving = False
        self._lower_hard_limit = -np.inf * q.mm
        self._upper_hard_limit = np.inf * q.mm

    def _set_position(self, position):
        direction = 0
        if self.position < position:
            direction = 1
        elif self.position > position:
            direction = -1

        if direction:
            self._moving = True
            while (direction * self._position) < (direction * position) and self._moving:
                self._position += direction * self.motion_velocity * MOVEMENT_TIME_STEPS
                sleep(MOVEMENT_TIME_STEPS.to(q.s).magnitude)
                if self._position < self._lower_hard_limit:
                    self._moving = False
                    self._position = self._lower_hard_limit
                    raise HardLimitError('hard-limit')
                if self._position > self._upper_hard_limit:
                    self._moving = False
                    self._position = self._upper_hard_limit
                    raise HardLimitError('hard-limit')
            if self._moving:
                self._position = position
        self._moving = False

    def _get_position(self):
        return self._position

    def _stop(self):
        self._moving = False

    def _set_motion_velocity(self, vel):
        self._motion_velocity = vel

    def _get_motion_velocity(self):
        return self._motion_velocity

    def _get_state(self):
        if self._moving:
            return 'moving'
        if self._position > self._lower_hard_limit and self._position < self._upper_hard_limit:
            return 'standby'
        return 'hard-limit'


class LinearMotor(_PositionMixin, base.LinearMotor):

    """A linear step motor dummy."""

    motion_velocity = Quantity(q.mm/q.s)

    def __init__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        base.LinearMotor.__init__(self)
        _PositionMixin.__init__(self)
        self.motion_velocity = 200000*q.mm/q.s

        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit


class ContinuousLinearMotor(LinearMotor, base.ContinuousLinearMotor):

    """A continuous linear motor dummy."""

    def __init__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        base.ContinuousLinearMotor.__init__(self)
        LinearMotor.__init__(self)
        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit

    def _set_velocity(self, vel):
        if vel.magnitude > 0:
            self.motion_velocity = vel
            if self['position'].upper:
                self.set_position(self['position'].upper)
            else:
                self.set_position(np.inf * q.mm)
        if vel.magnitude < 0:
            self.motion_velocity = vel
            if self['position'].lower:
                self.set_position(self['position'].lower)
            else:
                self.set_position(-np.inf * q.mm)
        if vel.magnitude == 0:
            self.stop()

    def _stop(self):
        self._moving = False


class RotationMotor(_PositionMixin, base.RotationMotor):

    """A rotational step motor dummy."""

    motion_velocity = Quantity(q.deg/q.s)

    def __init__(self, upper_hard_limit=None, lower_hard_limit=None):
        base.RotationMotor.__init__(self)
        _PositionMixin.__init__(self)
        self._position = 0 * q.deg
        self._lower_hard_limit = -np.inf * q.deg
        self._upper_hard_limit = np.inf * q.deg
        self.motion_velocity = 50000 * q.deg / q.s
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit


class ContinuousRotationMotor(RotationMotor,
                              base.ContinuousRotationMotor):

    """A continuous rotational step motor dummy."""

    def __init__(self, position=None, upper_hard_limit=None, lower_hard_limit=None):
        base.ContinuousRotationMotor.__init__(self)
        RotationMotor.__init__(self)
        if position:
            self._position = position
        if upper_hard_limit:
            self._upper_hard_limit = upper_hard_limit
        if lower_hard_limit:
            self._lower_hard_limit = lower_hard_limit

    def _set_velocity(self, vel):
        if vel.magnitude > 0:
            self.motion_velocity = vel
            if self['position'].upper:
                self.set_position(self['position'].upper)
            else:
                self.set_position(np.inf * q.deg)
        if vel.magnitude < 0:
            self.motion_velocity = vel
            if self['position'].lower:
                self.set_position(self['position'].lower)
            else:
                self.set_position(-np.inf * q.deg)
        if vel.magnitude == 0:
            self.stop()
