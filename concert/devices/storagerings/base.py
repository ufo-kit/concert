"""Storage Ring Device"""
from abc import abstractmethod

from concert.quantities import q
from concert.base import Quantity
from concert.devices.base import Device


class StorageRing(Device):

    """Read-only access to storage ring information.

    .. py:attribute:: current

        Ring current

    .. py:attribute:: energy

        Ring energy

    .. py:attribute:: lifetime

        Ring lifetime in hours
    """
    current = Quantity(q.mA, help="Current")
    energy = Quantity(q.MeV, help="Energy")
    lifetime = Quantity(q.hour, help="Expected lifetime")

    async def __ainit__(self):
        await super(StorageRing, self).__ainit__()

    @abstractmethod
    async def _get_current(self):
        ...

    @abstractmethod
    async def _get_energy(self):
        ...

    @abstractmethod
    async def _get_lifetime(self):
        ...
