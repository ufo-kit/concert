import random
import time
import quantities as q
from concert.devices.axes.base import Axis, ContinuousAxis, AxisMessage
from concert.devices.axes.base import AxisState


class DummyLimiter(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, value):
        return self.low < value < self.high


class DummyAxis(Axis):
    def __init__(self, calibration, limiter=None):
        super(DummyAxis, self).__init__(calibration)
        self._hard_limits = -100, 100
        if limiter is None:
            self._position = random.uniform(self._hard_limits[0],
                                            self._hard_limits[1])
        else:
            self._position = random.uniform(limiter.low, limiter.high)

        self._register('position',
                       self._get_position,
                       self._set_position,
                       q.m,
                       limiter)

    def _stop_real(self):
        pass

    def hard_position_limit_reached(self):
        return self._position <= self._hard_limits[0] or\
            self._position >= self._hard_limits[1]

    def _set_position(self, position):
        self._set_state(AxisState.MOVING)

        time.sleep(random.random() / 25.)

        if position < self._hard_limits[0]:
            self._position = self._hard_limits[0]
            self.send(AxisMessage.POSITION_LIMIT)
        elif position > self._hard_limits[1]:
            self._position = self._hard_limits[1]
            self.send(AxisMessage.POSITION_LIMIT)
        else:
            self._position = position
            self._set_state(AxisState.STANDBY)

    def _get_position(self):
        return self._position


class DummyContinuousAxis(ContinuousAxis):
    def __init__(self, position_calibration, velocity_calibration):
        super(DummyContinuousAxis, self).__init__(position_calibration,
                                                  velocity_calibration)
        self._position_hard_limits = -10, 10
        self._velocity_hard_limits = -100, 100
        self._position = 0
        self._velocity = 0

        self._register('position',
                       self._get_position,
                       self._set_position,
                       q.m)

        self._register('velocity',
                       self._get_velocity,
                       self._set_velocity,
                       q.m / q.s)

    def _stop_real(self):
        time.sleep(0.1)
        self._velocity = 0

    def _set_position(self, position):
        time.sleep(random.random() / 25.)

        self._position = position
        if self._position < self._position_hard_limits[0]:
            self._position = self._position_hard_limits[0]
        elif self._position > self._position_hard_limits[1]:
            self._position = self._position_hard_limits[1]

    def _get_position(self):
        return self._position

    def _set_velocity(self, velocity):
        self._set_state(AxisState.MOVING)

        time.sleep(random.random())
        self._velocity = velocity

        if self._velocity < self._velocity_hard_limits[0]:
            self._velocity = self._velocity_hard_limits[0]
            self.send(AxisMessage.VELOCITY_LIMIT)
        elif self._velocity > self._velocity_hard_limits[1]:
            self._velocity = self._velocity_hard_limits[1]
            self.send(AxisMessage.VELOCITY_LIMIT)
        else:
            self._set_state(AxisState.STANDBY)

    def _get_velocity(self):
        return self._velocity
