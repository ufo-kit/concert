"""Pumps."""

from concert.base import Quantity, State, check, AccessorNotImplementedError
from concert.quantities import q
from concert.async import async
from concert.devices.base import Device


class Pump(Device):

    """A pumping device."""

    state = State(default='standby')
    flow_rate = Quantity(q.l / q.s, help="Flow rate")

    def __init__(self):
        super(Pump, self).__init__()

    @async
    @check(source='standby', target='pumping')
    def start(self):
        """
        start()

        Start pumping.
        """
        self._start()

    @async
    @check(source='pumping', target='standby')
    def stop(self):
        """
        stop()

        Stop pumping.
        """
        self._stop()

    def _get_flow_rate(self):
        raise AccessorNotImplementedError

    def _set_flow_rate(self, flow_rate):
        raise AccessorNotImplementedError

    def _start(self):
        raise NotImplementedError

    def _stop(self):
        raise NotImplementedError
