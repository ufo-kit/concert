"""Motor Dummy."""
import random
from concert.base import HardLimitError
from concert.quantities import q
from concert.devices.motors import base


class _PositionMixin(object):
    def __init__(self):
        self.lower = -100 * q.mm
        self.upper = +100 * q.mm
        self._position = random.uniform(self.lower, self.upper)

    def _set_position(self, position):
        if position < self.lower:
            self._position = self.lower
            raise HardLimitError('hard-limit')
        elif position > self.upper:
            self._position = self.upper
            raise HardLimitError('hard-limit')
        else:
            self._position = position

    def _get_position(self):
        return self._position

    def _stop(self):
        pass


class _ContinuousMixin(object):
    def __init__(self):
        super(_ContinuousMixin, self).__init__()
        self.velocity_lower = -100 * q.mm / q.s
        self.velocity_upper = 100 * q.mm / q.s
        self._velocity = 0 * q.mm / q.s

    def _set_velocity(self, velocity):
        if velocity < self.velocity_lower:
            self._velocity = self.velocity_lower
            raise HardLimitError('hard-limit')
        elif velocity > self.velocity_upper:
            self._velocity = self.velocity_upper
            raise HardLimitError('hard-limit')
        else:
            self._velocity = velocity

    def _get_velocity(self):
        return self._velocity


class LinearMotor(base.LinearMotor, _PositionMixin):

    """A linear step motor dummy."""

    def __init__(self, position=None):
        super(LinearMotor, self).__init__()
        _PositionMixin.__init__(self)

        if position:
            self._position = position

    def _get_state(self):
        if self._position > self.lower and self._position < self.upper:
            return 'standby'

        return 'hard-limit'


class ContinuousLinearMotor(LinearMotor, base.ContinuousLinearMotor, _ContinuousMixin):

    """A continuous linear motor dummy."""

    def __init__(self):
        super(ContinuousLinearMotor, self).__init__()
        _ContinuousMixin.__init__(self)

    def _get_state(self):
        if self.velocity.magnitude != 0:
            return 'moving'

        return LinearMotor._get_state(self)

    def _stop(self):
        self._velocity = 0 * q.mm / q.s


class RotationMotor(base.RotationMotor, _PositionMixin):

    """A rotational step motor dummy."""

    def __init__(self):
        super(RotationMotor, self).__init__()
        _PositionMixin.__init__(self)
        self.lower = -float("Inf") * q.deg
        self.upper = float("Inf") * q.deg
        self._position = self._position.magnitude * q.deg

    def _get_state(self):
        return 'standby'


class ContinuousRotationMotor(RotationMotor,
                              base.ContinuousRotationMotor,
                              _ContinuousMixin):

    """A continuous rotational step motor dummy."""

    def __init__(self):
        super(ContinuousRotationMotor, self).__init__()
        _ContinuousMixin.__init__(self)
        self.velocity_lower = self.velocity_lower.magnitude * q.deg / q.s
        self.velocity_upper = self.velocity_upper.magnitude * q.deg / q.s
        self._velocity = self._velocity.magnitude * q.deg / q.s

    def _get_state(self):
        if abs(self.velocity) > 1e-3 * q.deg / q.s:
            return 'moving'

        return RotationMotor._get_state(self)

    def _stop(self):
        self._velocity = 0 * q.deg / q.s
