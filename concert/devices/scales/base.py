"""Base scales module for implementing scales."""

from concert.devices.base import Device
from concert.base import Quantity, AccessorNotImplementedError
from concert.quantities import q
from concert.async import async


class WeightError(Exception):

    """Class for weighing errors."""

    pass


class Scales(Device):

    """Base scales class."""
    weight = Quantity(q.g, help="Weighted mass")

    def __init__(self):
        super(Scales, self).__init__()

    def _get_weight(self):
        raise AccessorNotImplementedError


class TarableScales(Scales):

    """Scales which can be tared."""

    def __init__(self):
        super(TarableScales, self).__init__()

    @async
    def tare(self):
        """Tare the scales."""
        self._tare()

    def _tare(self):
        raise AccessorNotImplementedError
