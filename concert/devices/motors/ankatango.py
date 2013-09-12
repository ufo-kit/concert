"""
Tango motors with ANKA specific interfaces.
"""
import time
import logbook
from concert.devices.motors import base

LOG = logbook.Logger(__name__)

try:
    import PyTango
except ImportError:
    LOG.warn("PyTango is not installed.")


SLEEP_TIME = 0.01
# Yes, it is really THAT slow!
# TODO: when RATO is not used reconsider, it might get faster.
SLOW_SLEEP_TIME = 1.0


class Motor(base.Motor):

    """A motor based on ANKA Tango motor interface."""

    def __init__(self, device, calibration=None):
        super(Motor, self).__init__(calibration=calibration,
                                    in_hard_limit=self._in_hard_limit)
        self._device = device

    def _in_hard_limit(self):
        return self._device.BackwardLimitSwitch or \
            self._device.ForwardLimitSwitch

    def _get_state(self):
        tango_state = self._device.state()
        if tango_state == PyTango.DevState.MOVING:
            state = self.MOVING
        elif tango_state == PyTango.DevState.STANDBY:
            state = self.STANDBY
        else:
            state = self.NA

        return state

    def _set_position(self, position):
        self._device.position = position

        time.sleep(SLOW_SLEEP_TIME)

        while self._get_state() == base.Motor.MOVING:
            time.sleep(SLEEP_TIME)

    def _get_position(self):
        return self._device.position

    def _stop(self):
        self._device.Stop()

        while self._device.state() == PyTango.DevState.RUNNING:
            time.sleep(SLEEP_TIME)

    def _home(self):
        pass
