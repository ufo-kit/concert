"""A dummy pump."""

from concert.base import transition
from concert.devices.pumps import base
from concert.quantities import q


class Pump(base.Pump):

    """A dummy pump."""

    def __init__(self):
        super(Pump, self).__init__()
        self._flow_rate = 0 * q.l / q.s

    @transition(target='pumping')
    async def _start(self):
        pass

    @transition(target='standby')
    async def _stop(self):
        pass

    async def _set_flow_rate(self, flow_rate):
        self._flow_rate = flow_rate

    async def _get_flow_rate(self):
        return self._flow_rate
