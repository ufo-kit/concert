import numpy
import time
import quantities as q
from control.devices.axes.axis import Axis, ContinuousAxis
from control.devices.axes.axis import AxisState, ContinuousAxisState


class DummyAxis(Axis):
    def __init__(self, connection, calibration, position_limit=None):
        super(DummyAxis, self).__init__(connection,
                                        calibration,
                                        position_limit)
        self._hard_limits = -100, 100
        self._position = 0

        self._register('position',
                       self._get_position,
                       self._set_position,
                       q.m)

    def _stop_real(self):
        pass

    def _set_position(self, position):
        self._set_state(AxisState.MOVING)

        time.sleep(numpy.random.random() / 2.)

        if position < self._hard_limits[0]:
            self._position = self._hard_limits[0]
            self._set_state(AxisState.POSITION_LIMIT)
        elif position > self._hard_limits[1]:
            self._position = self._hard_limits[1]
            self._set_state(AxisState.POSITION_LIMIT)
        else:
            self._position = position
            self._set_state(AxisState.STANDBY)

    def _get_position(self):
        return self._position

    def _is_hard_position_limit_reached(self):
        return self._position <= self._hard_limits[0] or \
               self._position >= self._hard_limits[1]


class DummyContinuousAxis(ContinuousAxis):
    def __init__(self, connection, position_calibration, velocity_calibration,
                 position_limit=None, velocity_limit=None):

        super(DummyContinuousAxis, self).__init__(connection,
                                                  position_calibration,
                                                  velocity_calibration,
                                                  position_limit,
                                                  velocity_limit)
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
        time.sleep(0.5)
        self._velocity = 0

    def _set_position(self, position):
        time.sleep(numpy.random.random() / 2.)

        self._position = position
        if self._position < self._position_hard_limits[0]:
            self._position = self._position_hard_limits[0]
        elif self._position > self._position_hard_limits[1]:
            self._position = self._position_hard_limits[1]

    def _get_position(self):
        return self._position

    def _is_hard_position_limit_reached(self):
        return self._position <= self._position_hard_limits[0] or \
               self._position >= self._position_hard_limits[1]

    def _set_velocity(self, velocity):
        self._set_state(AxisState.MOVING)

        time.sleep(numpy.random.random())
        self._velocity = velocity

        if self._velocity < self._velocity_hard_limits[0]:
            self._velocity = self._velocity_hard_limits[0]
            self._set_state(ContinuousAxisState.VELOCITY_LIMIT)
        elif self._velocity > self._velocity_hard_limits[1]:
            self._velocity = self._velocity_hard_limits[1]
            self._set_state(ContinuousAxisState.VELOCITY_LIMIT)
        else:
            self._set_state(AxisState.STANDBY)

    def _get_velocity(self):
        return self._velocity

    def _is_hard_velocity_limit_reached(self):
        return self._position <= self._velocity_hard_limits[0] or \
               self._position >= self._velocity_hard_limits[1]
