"""Storage Ring Device"""
from concert.quantities import q
from concert.base import Quantity, AccessorNotImplementedError
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

    async def _get_current(self):
        raise AccessorNotImplementedError

    async def _get_energy(self):
        raise AccessorNotImplementedError

    async def _get_lifetime(self):
        raise AccessorNotImplementedError
