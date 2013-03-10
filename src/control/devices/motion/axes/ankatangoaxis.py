'''
Created on Mar 5, 2013

@author: farago
'''
from control.devices.motion.axes.axis import Axis
import time
import PyTango
from control.devices.motion.axes.axis import AxisState
from control.devices.device import State
from control.devices.device import UnknownStateError


SLEEP_TIME = 0.005
SLOW_SLEEP_TIME = 0.5


class ANKATangoDiscreteAxis(Axis):
    def __init__(self, connection, calibration, position_limit=None):
        super(ANKATangoDiscreteAxis, self).__init__(connection, calibration,
                                                position_limit) 
        
    @property
    def state(self):
        tango_state = self._connection.tango_device.state()
        if tango_state == PyTango.DevState.MOVING:
            return AxisState.MOVING
        elif tango_state == PyTango.DevState.ALARM:
            return AxisState.POSITION_LIMIT
        elif tango_state == PyTango.DevState.STANDBY:
            return AxisState.STANDBY
        elif tango_state == PyTango.DevState.FAULT:
            return State.ERROR
        else:
            raise UnknownStateError(tango_state)
        
    def _set_position_real(self, position):
        self._connection.tango_device.write_attribute("position", position)
        time.sleep(SLOW_SLEEP_TIME)
        while self.state == AxisState.MOVING:
            time.sleep(SLEEP_TIME)
        
    def _get_position_real(self):
        return self._connection.tango_device.read_attribute("position").value
    
    def _stop_real(self):
        self._connection.tango_device.command_inout("Stop")
        while self._connection.tango_device.state() == PyTango.DevState.RUNNING:
            time.sleep(SLEEP_TIME)
    
    def home(self):
        pass
    
    def is_hard_position_limit_reached(self):
        return self._connection.tango_device.BackwardLimitSwitch or\
                self._connection.tango_device.ForwardLimitSwitch
