"""Motor Dummy."""
import random
from concert.base import HardLimitError
from concert.quantities import q
from concert.devices.motors import base


class _PositionMixin(object):
    def __init__(self):
        self.lower = -100 * q.count
        self.upper = +100 * q.count
        self._position = random.uniform(self.lower, self.upper)

    def _set_position(self, position):
        if position < self.lower:
            self._position = self.lower
        elif position > self.upper:
            self._position = self.upper
        else:
            self._position = position

    def _get_position(self):
        return self._position

    def _stop(self):
        pass


class _ContinuousMixin(object):
    def __init__(self):
        super(_ContinuousMixin, self).__init__()
        self.velocity_lower = -100 * q.count
        self.velocity_upper = 100 * q.count
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


class LinearMotor(base.LinearMotor, _PositionMixin):

    def __init__(self, position=None):
        super(LinearMotor, self).__init__()
        _PositionMixin.__init__(self)
        self['position'].conversion = lambda x: x / q.mm * q.count

        if position:
            self._position = position

    def check_state(self):
        if self._position > self.lower and self._position < self.upper:
            return 'standby'

        raise HardLimitError('hard-limit')


class ContinuousLinearMotor(LinearMotor, base.ContinuousLinearMotor, _ContinuousMixin):

    def __init__(self):
        super(ContinuousLinearMotor, self).__init__()
        _ContinuousMixin.__init__(self)
        self['velocity'].conversion = lambda x: x / q.mm * q.s * q.count

    def check_state(self):
        if self.velocity.magnitude != 0:
            return 'moving'

        return LinearMotor.check_state(self)

    def _stop(self):
        self._velocity = 0 * q.count


class RotationMotor(base.RotationMotor, _PositionMixin):

    def __init__(self):
        super(RotationMotor, self).__init__()
        _PositionMixin.__init__(self)
        self['position'].conversion = lambda x: x / q.deg * q.count
        self.lower = -float("Inf") * q.count
        self.upper = float("Inf") * q.count

    def check_state(self):
        return 'standby'


class ContinuousRotationMotor(RotationMotor,
                              base.ContinuousRotationMotor,
                              _ContinuousMixin):

    def __init__(self):
        super(ContinuousRotationMotor, self).__init__()
        _ContinuousMixin.__init__(self)
        self['velocity'].conversion = lambda x: x / q.deg * q.s * q.count

    def check_state(self):
        if self.velocity.magnitude != 0:
            return 'moving'

        return RotationMotor.check_state(self)

    def _stop(self):
        self._velocity = 0 * q.count
