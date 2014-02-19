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
    current = Quantity(q.mA, help="Current")
    energy = Quantity(q.MeV, help="Energy")
    lifetime = Quantity(q.h, help="Expected lifetime")

    def __init__(self):
        super(StorageRing, self).__init__()

    def _get_current(self):
        raise NotImplementedError

    def _get_energy(self):
        raise NotImplementedError

    def _get_lifetime(self):
        raise NotImplementedError
