"""Base scales module for implementing scales."""
from abc import abstractmethod

from concert.coroutines.base import background
from concert.devices.base import Device
from concert.base import Quantity
from concert.quantities import q


class WeightError(Exception):

    """Class for weighing errors."""

    pass


class Scales(Device):

    """Base scales class."""
    weight = Quantity(q.g, help="Weighted mass")

    async def __ainit__(self):
        await super(Scales, self).__ainit__()

    @abstractmethod
    async def _get_weight(self):
        ...


class TarableScales(Scales):

    """Scales which can be tared."""

    async def __ainit__(self):
        await super(TarableScales, self).__ainit__()

    @background
    async def tare(self):
        """
        tare()

        Tare the scales.
        """
        await self._tare()

    @abstractmethod
    async def _tare(self):
        ...
