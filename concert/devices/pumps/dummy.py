"""A dummy pump."""

from concert.devices.pumps import base
from concert.quantities import q


class Pump(base.Pump):

    """A dummy pump."""

    def __init__(self):
        super(Pump, self).__init__()
        self._flow_rate = 0 * q.count

    def _start(self):
        self._set_state(Pump.PUMPING)

    def _stop(self):
        self._set_state(Pump.STANDBY)

    def _set_flow_rate(self, flow_rate):
        self._flow_rate = flow_rate

    def _get_flow_rate(self):
        return self._flow_rate
