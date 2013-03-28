'''
Created on Mar 5, 2013

@author: farago
'''
import quantities as q
from concert.base import Device, Parameter


class StorageRing(Device):
    def __init__(self):
        params = [Parameter('current',
                            self._get_current,
                            unit=q.mA,
                            doc="Current of the ring"),
                  Parameter('energy',
                            self._get_energy,
                            unit=q.MeV,
                            doc="Energy of the ring"),
                  Parameter('lifetime',
                            self._get_lifetime,
                            unit=q.h,
                            doc="Lifetime of the ring")]

        super(StorageRing, self).__init__(params)

    def _get_current(self):
        raise NotImplementedError

    def _get_energy(self):
        raise NotImplementedError

    def _get_lifetime(self):
        raise NotImplementedError
