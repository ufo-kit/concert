"""Base scales module for implementing scales."""

from concert.devices.base import Device
from concert.base import Parameter
from concert.quantities import q
from concert.helpers import async


class WeightError(Exception):

    """Class for weighing errors."""

    pass


class Scales(Device):

    """Base scales class."""

    def __init__(self, calibration):
        params = [Parameter("weight", fget=self._get_calibrated_weight,
                            unit=q.g, doc="Weight")]
        super(Scales, self).__init__(parameters=params)
        self._calibration = calibration

    def _get_calibrated_weight(self):
        return self._calibration.to_user(self._get_weight())

    def _get_weight(self):
        raise NotImplementedError


class TarableScales(Scales):

    """Scales which can be tared."""

    def __init__(self, calibration):
        super(TarableScales, self).__init__(calibration)

    @async
    def tare(self):
        """Tare the scales."""
        self._tare()

    def _tare(self):
        raise NotImplementedError
