"""Pumps."""

from concert.base import Quantity, State, transition
from concert.quantities import q
from concert.async import async
from concert.devices.base import Device


class Pump(Device):

    """
    Every pump has a calibration for proper unit conversion and a flow
    rate limit determined by *lower* and *upper*.
    """

    state = State(default='standby')
    flow_rate = Quantity(unit=q.l / q.s, conversion=lambda x: x * q.s / q.l)

    def __init__(self):
        super(Pump, self).__init__()

    @async
    @transition(source='standby', target='pumping')
    def start(self):
        """
        start()

        Start pumping.
        """
        self._start()

    @async
    @transition(source='pumping', target='standby')
    def stop(self):
        """
        stop()

        Stop pumping.
        """
        self._stop()

    def _get_flow_rate(self):
        raise NotImplementedError

    def _set_flow_rate(self, flow_rate):
        raise NotImplementedError

    def _start(self):
        raise NotImplementedError

    def _stop(self):
        raise NotImplementedError
