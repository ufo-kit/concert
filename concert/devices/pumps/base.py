"""Pumps."""

from concert.base import Quantity
from concert.quantities import q
from concert.async import async
from concert.fsm import State, transition
from concert.devices.base import Device


class Pump(Device):

    """
    Every pump has a calibration for proper unit conversion and a flow
    rate limit determined by *lower* and *upper*.
    """

    state = State(default='standby')

    def __init__(self, calibration):
        params = [Quantity('flow_rate',
                           fget=self._get_calibrated_flow_rate,
                           fset=self._set_calibrated_flow_rate,
                           unit=q.l / q.s, doc="Pump flow rate.")]
        super(Pump, self).__init__(params)
        self._calibration = calibration

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

    def _get_calibrated_flow_rate(self):
        return self._calibration.to_user(self._get_flow_rate())

    def _set_calibrated_flow_rate(self, flow_rate):
        self._set_flow_rate(self._calibration.to_device(flow_rate))

    def _get_flow_rate(self):
        raise NotImplementedError

    def _set_flow_rate(self, flow_rate):
        raise NotImplementedError

    def _start(self):
        raise NotImplementedError

    def _stop(self):
        raise NotImplementedError
