"""A dummy pump."""

from concert.devices.pumps import base
from concert.devices.motors.base import LinearCalibration
from concert.quantities import q


class Pump(base.Pump):

    """Dummy pump class with some parameters."""

    def __init__(self):
        super(Pump, self).__init__(LinearCalibration(q.count * q.min / q.ml,
                                                     0 * q.ml / q.min),
                                   0 * q.ml / q.min, 1000 * q.ml / q.min)
        self._flow_rate = 0 * q.count
        self._set_state(self.STANDBY)

    def _start(self):
        self._set_state(Pump.PUMPING)

    def _stop(self):
        self._set_state(Pump.STANDBY)

    def _set_flow_rate(self, flow_rate):
        self._flow_rate = flow_rate

    def _get_flow_rate(self):
        return self._flow_rate
