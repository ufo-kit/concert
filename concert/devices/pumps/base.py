"""Pumps."""

from concert.base import Parameter
from concert.quantities import q
from concert.helpers import async
from concert.devices.base import Device


class Pump(Device):

    """
    Every pump has a calibration for proper unit conversion and a flow
    rate limit determined by *lower* and *upper*.
    """

    PUMPING = "pumping"
    STANDBY = "standby"

    def __init__(self, calibration):
        params = [Parameter('flow_rate',
                            fget=self._get_calibrated_flow_rate,
                            fset=self._set_calibrated_flow_rate,
                            unit=q.l / q.s, doc="Pump flow rate.")]
        super(Pump, self).__init__(params)
        self._calibration = calibration
        self._states = self._states.union(set([self.STANDBY, self.PUMPING]))

    @async
    def start(self):
        """
        start()

        Start pumping.
        """
        self._start()

    @async
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
