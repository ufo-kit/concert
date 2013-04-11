"""
Tango motors with ANKA specific interfaces.
"""
import time
import logbook
from concert.devices.motors.base import Motor, MotorMessage

log = logbook.Logger(__name__)

try:
    import PyTango
except ImportError:
    log.warn("PyTango is not installed.")


SLEEP_TIME = 0.005
# Yes, it is really THAT slow!
# TODO: when RATO is not used reconsider, it might get faster.
SLOW_SLEEP_TIME = 1.0


class Discrete(Motor):
    """A motor based on ANKA Tango motor interface."""
    def __init__(self, connection, calibration, position_limit=None):
        super(Discrete, self).__init__(calibration)
        self._connection = connection
        self._state = self._determine_state()

    def _query_state(self):
        tango_state = self._connection.tango_device.state()
        if tango_state == PyTango.DevState.MOVING:
            current = Motor.MOVING
        elif tango_state == PyTango.DevState.STANDBY:
            current = Motor.STANDBY
        else:
            current = None
        
        return current

    def _set_position(self, position):
        self._connection.tango_device.write_attribute("position", position)
        time.sleep(SLOW_SLEEP_TIME)
        # while self.state == MotorState.MOVING:
        #     time.sleep(SLEEP_TIME)
        if self.hard_position_limit_reached():
            self.send(MotorMessage.POSITION_LIMIT)

    def _get_position(self):
        return self._connection.tango_device.read_attribute("position").value

    def _stop(self):
        device = self._connection.tango_device
        device.command_inout("Stop")

        while device.state() == PyTango.DevState.RUNNING:
            time.sleep(SLEEP_TIME)

    def _home(self):
        pass

    def hard_position_limit_reached(self):
        return self._connection.tango_device.BackwardLimitSwitch or\
            self._connection.tango_device.ForwardLimitSwitch
