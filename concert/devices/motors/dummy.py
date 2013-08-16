"""Motor Dummy."""
import random
from concert.quantities import q
from concert.devices.motors import base
from concert.devices.base import LinearCalibration


class DummyLimiter(object):

    """Dummy Limiter class implementation."""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, value):
        return self.low < value < self.high


class Motor(base.Motor):

    """Dummy Motor class implementation."""

    def __init__(self, calibration=None,
                 limiter=None, position=None, hard_limits=None):
        super(Motor, self).__init__(calibration, limiter)
        if hard_limits is None:
            self._hard_limits = -100, 100
        else:
            self._hard_limits = hard_limits
        if not limiter:
            self._position = random.uniform(self._hard_limits[0],
                                            self._hard_limits[1])
        else:
            self._position = position

    def in_hard_limit(self):
        return self._position < self._hard_limits[0] or not\
            self._position < self._hard_limits[1]

    def _stop_real(self):
        pass

    def _set_position(self, position):
        if position < self._hard_limits[0]:
            self._position = self._hard_limits[0]
        elif not position < self._hard_limits[1]:
            # We do this funny comparison because pint is able to compare
            # "position < something" but not the other way around. See
            # https://github.com/hgrecco/pint/issues/40
            self._position = self._hard_limits[1]
        else:
            self._position = position

    def _get_position(self):
        return self._position


class ContinuousMotor(base.ContinuousMotor):

    """Dummy ContinuousMotor class implementation."""

    def __init__(self, position_calibration, velocity_calibration):
        super(ContinuousMotor, self).__init__(position_calibration,
                                              velocity_calibration)
        self._position_hard_limits = -10, 10
        self._velocity_hard_limits = -100, 100
        self._position = 0
        self._velocity = 0

    def _stop_real(self):
        self._velocity = 0

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
