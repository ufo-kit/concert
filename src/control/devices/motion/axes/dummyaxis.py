'''
Created on Mar 5, 2013

@author: farago
'''
import numpy
import time
from control.devices.motion.axes.axis import Axis
from control.devices.motion.axes.axis import ContinuousAxis
from control.devices.motion.axes.axis import AxisState
from control.events import generator as eventgenerator
from control.events.event import Event
from control.events import type as eventtype
from control.devices.motion.axes.axis import ContinuousAxisState


class DummyAxis(Axis):
    def __init__(self, connection, calibration, position_limit=None):
        super(DummyAxis, self).__init__(connection,
                                        calibration,
                                        position_limit)
        self._hard_limits = -100, 100
        self._position = 0
        self._state = AxisState.STANDBY

    @property
    def state(self):
        return self._state

    def _set_position_real(self, position):
        self._state = AxisState.MOVING
        eventgenerator.fire(Event(eventtype.StateChangeEvent.STATE,
                                      self, AxisState.MOVING))
        time.sleep(numpy.random.random())
        self._position = position
        if self._position < self._hard_limits[0]:
            self._position = self._hard_limits[0]
            self._state = AxisState.POSITION_LIMIT
            eventgenerator.fire(Event(eventtype.StateChangeEvent.STATE,
                                      self, self._state))
        elif self._position > self._hard_limits[1]:
            self._position = self._hard_limits[1]
            self._state = AxisState.POSITION_LIMIT
            eventgenerator.fire(Event(eventtype.StateChangeEvent.STATE,
                                      self, self._state))
        else:
            self._state = AxisState.STANDBY
            eventgenerator.fire(Event(eventtype.StateChangeEvent.STATE,
                                      self, self._state))

    def _get_position_real(self):
        return self._position

    def _is_hard_position_limit_reached(self):
        return self._position <= self._hard_limits[0] or\
                self._position >= self._hard_limits[1]


class DummyContinuousAxis(ContinuousAxis):
    def __init__(self, connection, position_calibration, velocity_calibration,
                                position_limit=None, velocity_limit=None):
        super(DummyContinuousAxis, self).__init__(connection,
                          position_calibration, velocity_calibration,
                          position_limit, velocity_limit)
        self._position_hard_limits = -10, 10
        self._velocity_hard_limits = -100, 100
        self._position = 0
        self._velocity = 0

    def _stop_real(self):
        time.sleep(0.5)
        self._velocity = 0

    def _set_position_real(self, position):
        time.sleep(numpy.random.random())
        self._position = position
        if self._position < self._position_hard_limits[0]:
            self._position = self._position_hard_limits[0]
        elif self._position > self._position_hard_limits[1]:
            self._position = self._position_hard_limits[1]

    def _get_position_real(self):
        return self._position

    def _is_hard_position_limit_reached(self):
        return self._position <= self._position_hard_limits[0] or\
                self._position >= self._position_hard_limits[1]

    def _set_velocity_real(self, velocity):
        self._state = AxisState.MOVING
        time.sleep(numpy.random.random())
        self._velocity = velocity
        if self._velocity < self._velocity_hard_limits[0]:
            self._velocity = self._velocity_hard_limits[0]
            self._state = ContinuousAxisState.VELOCITY_LIMIT
            eventgenerator.fire(Event(eventtype.StateChangeEvent.STATE,
                                      self, self._state))
        elif self._velocity > self._velocity_hard_limits[1]:
            self._velocity = self._velocity_hard_limits[1]
            self._state = ContinuousAxisState.VELOCITY_LIMIT
            eventgenerator.fire(Event(eventtype.StateChangeEvent.STATE,
                                      self, self._state))
        else:
            self._state = AxisState.STANDBY
            eventgenerator.fire(Event(eventtype.StateChangeEvent.STATE,
                                      self, self._state))

    def _get_velocity_real(self):
        return self._velocity

    def _is_hard_velocity_limit_reached(self):
        return self._position <= self._velocity_hard_limits[0] or\
                self._position >= self._velocity_hard_limits[1]
