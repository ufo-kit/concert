"""Motor Dummy."""
import random
from concert.quantities import q
from concert.devices.motors import base


class Motor(base.Motor):

    """Dummy Motor class implementation."""

    def __init__(self, calibration=None, position=None, hard_limits=None):
        super(Motor, self).__init__(calibration=calibration,
                                    in_hard_limit=self._in_hard_limit)

        if hard_limits:
            self.lower, self.upper = hard_limits
        else:
            self.lower, self.upper = -100, 100
        self.lower = self.lower * q.count
        self.upper = self.upper * q.count

        if position is not None:
            self._position = position
        else:
            self._position = random.uniform(self.lower, self.upper)

    def _in_hard_limit(self):
        return self._position < self.lower or not self._position < self.upper

    def _stop_real(self):
        pass

    def _set_position(self, position):
        if position < self.lower:
            self._position = self.lower
        elif not position < self.upper:
            # We do this funny comparison because pint is able to compare
            # "position < something" but not the other way around. See
            # https://github.com/hgrecco/pint/issues/40
            self._position = self.upper
        else:
            self._position = position

    def _get_position(self):
        return self._position


class ContinuousMotor(base.ContinuousMotor):

    """Dummy ContinuousMotor class implementation."""

    def __init__(self, position_calibration, velocity_calibration):
        super(ContinuousMotor, self).__init__(position_calibration,
                                              velocity_calibration)
        self._position_hard_limits = -10 * q.count, 10 * q.count
        self._velocity_hard_limits = -100 * q.count, 100 * q.count
        self._position = 0 * q.count
        self._velocity = 0 * q.count

    def _stop_real(self):
        self._velocity = 0 * q.count

    def _set_position(self, position):
        self._position = position
        if self._position < self._position_hard_limits[0]:
            self._position = self._position_hard_limits[0]
        elif not self._position < self._position_hard_limits[1]:
            self._position = self._position_hard_limits[1]

    def _get_position(self):
        return self._position

    def _set_velocity(self, velocity):
        self._velocity = velocity

        if self._velocity < self._velocity_hard_limits[0]:
            self._velocity = self._velocity_hard_limits[0]
            # self.send(MotorMessage.VELOCITY_LIMIT)
        elif not self._velocity < self._velocity_hard_limits[1]:
            self._velocity = self._velocity_hard_limits[1]
            # self.send(MotorMessage.VELOCITY_LIMIT)

    def _get_velocity(self):
        return self._velocity
