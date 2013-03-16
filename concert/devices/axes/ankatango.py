"""
Tango axes with ANKA specific interfaces.
"""
import time
import quantities as pq
import logbook
from concert.devices.axes.base import Axis, AxisState, AxisMessage
from concert.devices.base import UnknownStateError
from threading import Thread

log = logbook.Logger(__name__)

try:
    import PyTango
except ImportError:
    log.warn("PyTango is not installed.")


SLEEP_TIME = 0.005
# Yes, it is really THAT slow!
# TODO: when RATO is not used reconsider, it might get faster.
SLOW_SLEEP_TIME = 1.0


class ANKATangoDiscreteAxis(Axis):
    """Tango device that ... need ... more ... information."""
    def __init__(self, connection, calibration, position_limit=None):
        super(ANKATangoDiscreteAxis, self).__init__(calibration)
        self._connection = connection
        self._register("position", self._get_position_real,
                       self._set_position_real, pq.mm)
        self._state = self._determine_state()
        # State polling.
        self._poller = Thread(target=self._poll_state)
        self._poller.daemon = True
        self._poller.start()

    def _determine_state(self):
        tango_state = self._connection.tango_device.state()
        if tango_state == PyTango.DevState.MOVING:
            current = AxisState.MOVING
        elif tango_state == PyTango.DevState.STANDBY:
            current = AxisState.STANDBY
        else:
            raise UnknownStateError(tango_state)

        return current

    def _poll_state(self):
        while True:
            current = self._determine_state()
            if current != self._state:
                self._set_state(current)
            time.sleep(SLEEP_TIME)

    def _set_position_real(self, position):
        self._connection.tango_device.write_attribute("position", position)
        time.sleep(SLOW_SLEEP_TIME)
        while self.state == AxisState.MOVING:
            time.sleep(SLEEP_TIME)
        if self.hard_position_limit_reached():
            self.send(AxisMessage.POSITION_LIMIT)

    def _get_position_real(self):
        return self._connection.tango_device.read_attribute("position").value

    def _stop_real(self):
        device = self._connection.tango_device
        device.command_inout("Stop")

        while device.state() == PyTango.DevState.RUNNING:
            time.sleep(SLEEP_TIME)

    def home(self):
        pass

    def hard_position_limit_reached(self):
        return self._connection.tango_device.BackwardLimitSwitch or\
            self._connection.tango_device.ForwardLimitSwitch
