"""Storage Ring Device"""
from concert.quantities import q
from concert.base import Parameter
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

    def __init__(self):
        params = [Parameter('current',
                            fget=self._get_current,
                            unit=q.mA,
                            doc="Current of the ring"),
                  Parameter('energy',
                            fget=self._get_energy,
                            unit=q.MeV,
                            doc="Energy of the ring"),
                  Parameter('lifetime',
                            fget=self._get_lifetime,
                            unit=q.h,
                            doc="Lifetime of the ring")]

        super(StorageRing, self).__init__(params)

    def _get_current(self):
        raise NotImplementedError

    def _get_energy(self):
        raise NotImplementedError

    def _get_lifetime(self):
        raise NotImplementedError
