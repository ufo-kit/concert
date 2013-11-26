"""Motor Dummy."""
import random
from concert.quantities import q
from concert.devices.motors import base


class PositionMotorMixin(base.PositionMixin):
    def __init__(self):
        super(PositionMotorMixin, self).__init__()
        self.lower = -100 * q.count
        self.upper = +100 * q.count
        self._position = random.uniform(self.lower, self.upper)

    def _in_hard_limit(self):
        return self._position <= self.lower or self._position >= self.upper

    def _set_position(self, position):
        if position < self.lower:
            self._position = self.lower
        elif position > self.upper:
            self._position = self.upper
        else:
            self._position = position

    def _get_position(self):
        return self._position

    def _stop_real(self):
        pass


class ContinuousMotorMixin(base.ContinuousMixin):
    def __init__(self):
        super(ContinuousMotorMixin, self).__init__()
        self.velocity_lower = -100 * q.count / q.s
        self.velocity_upper = 100 * q.count / q.s
        self._velocity = 0 * q.count / q.s

    def _in_velocity_hard_limit(self):
        return self._velocity <= self.velocity_lower or \
            self._velocity >= self.velocity_upper

    def _stop_real(self):
        self._velocity = 0 * q.count

    def _set_velocity(self, velocity):
        if velocity < self.velocity_lower:
            self._velocity = self.velocity_lower
        elif velocity > self.velocity_upper:
            self._velocity = self.velocity_upper
        else:
            self._velocity = velocity

    def _get_velocity(self):
        return self._velocity


class Motor(base.Motor, PositionMotorMixin):

    def __init__(self, position=None, hard_limits=None):
        super(Motor, self).__init__()
        self['position'].conversion = lambda x: x / q.m * q.count

        if hard_limits:
            self.lower, self.upper = hard_limits

        if position:
            self._position = position


class ContinuousMotor(Motor, base.ContinuousMotor, ContinuousMotorMixin):

    def __init__(self):
        super(ContinuousMotor, self).__init__()
        self['velocity'].conversion = lambda x: x / q.m * q.count


class RotationMotor(base.RotationMotor, PositionMotorMixin):

    def __init__(self):
        super(RotationMotor, self).__init__()
        self['position'].conversion = lambda x: x / q.deg * q.count


class ContinuousRotationMotor(RotationMotor,
                              base.ContinuousRotationMotor,
                              ContinuousMotorMixin):

    def __init__(self):
        super(ContinuousRotationMotor, self).__init__()
        self['velocity'].conversion = lambda x: x / q.deg * q.count
