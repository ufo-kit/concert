"""Storage Ring Device"""
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
    current = Quantity(unit=q.mA)
    energy = Quantity(unit=q.MeV)
    lifetime = Quantity(unit=q.h)

    def __init__(self):
        super(StorageRing, self).__init__()

    def _get_current(self):
        raise NotImplementedError

    def _get_energy(self):
        raise NotImplementedError

    def _get_lifetime(self):
        raise NotImplementedError
