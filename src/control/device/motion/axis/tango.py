'''
Created on Feb 19, 2013

@author: farago
'''
import os
import PyTango
import time
from axis import DiscretelyMovable

SLEEP_TIME = 0.5


class TangoLinearMotor(DiscretelyMovable):
    """Class representing a linear motor communicating via Tango."""
    def __init__(self, min_limit, max_limit, name, tango_host=None,
                                                            tango_port=None):
        super(TangoLinearMotor, self).__init__()

        # Set the host and port for connecting to the Tango database.
        # TODO: check if there is a way to adjust the host in PyTango.
        if tango_host is not None and tango_port is not None:
            os.environ["TANGO_HOST"] = "%s:%d" % (tango_host, tango_port)

        # Get the tango motor.
        self._name = name
        self._tango_motor = PyTango.DeviceProxy(name)

    @property
    def name(self):
        return self._name

    def set_motor_position(self, param, value):
        self._tango_motor.write_attribute("position", value)

    def get_limits(self):
        step = 2

        # Forward limit.
        while not self._tango_motor.backwardlimitswitch:
            self.position.value += step
            # Take the remote call delay into account.
            time.sleep(SLEEP_TIME)
            while self._tango_motor.state() == PyTango.DevState.MOVING:
                time.sleep(SLEEP_TIME)
        # The value differs from the RangedParameter value.
        forward_limit = self._tango_motor.position

        # Backward limit.
        while not self._tango_motor.forwardlimitswitch:
            self.position.value -= step
            time.sleep(SLEEP_TIME)
            while self._tango_motor.state() == PyTango.DevState.MOVING:
                time.sleep(SLEEP_TIME)
        backward_limit = self._tango_motor.position

        return backward_limit, forward_limit


if __name__ == '__main__':
    motor = TangoLinearMotor(-1000, 1000, "iss/tomotable/m_elyafoc")
    print motor.get_limits()
