"""Motor Dummy."""
import random
from concert.quantities import q
from concert.devices.motors import base


class Motor(base.Motor):

    """Dummy Motor class implementation."""

    def __init__(self, position=None, hard_limits=None):
        super(Motor, self).__init__()

        if hard_limits:
            self.lower, self.upper = hard_limits
        else:
            self.lower, self.upper = -100 * q.count, 100 * q.count

        if position is not None:
            self._position = position
        else:
            self._position = random.uniform(self.lower, self.upper)

    def _in_hard_limit(self):
        return self._position <= self.lower or self._position >= self.upper

    def _stop_real(self):
        pass

    def _set_position(self, position):
        if position < self.lower:
            self._position = self.lower
        elif position > self.upper:
            self._position = self.upper
        else:
            self._position = position

    def _get_position(self):
        return self._position


class ContinuousMotor(base.ContinuousMotor, Motor):

    """Dummy ContinuousMotor class implementation."""

    def __init__(self):
        super(ContinuousMotor, self).__init__()

        self.velocity_lower = -100 * q.count / q.s
        self.velocity_upper = 100 * q.count / q.s
        self._position = 0 * q.count
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
