'''
Created on Mar 5, 2013

@author: farago
'''
import quantities as q
from concert.base import Device, Parameter


class StorageRing(Device):
    def __init__(self):
        def do_nothing(value):
            pass

        params = [Parameter('current',
                            fget=self._get_current,
                            fset=do_nothing,
                            unit=q.mA,
                            doc="Current of the ring",
                            owner_only=True),
                  Parameter('energy',
                            fget=self._get_energy,
                            fset=do_nothing,
                            unit=q.MeV,
                            doc="Energy of the ring",
                            owner_only=True),
                  Parameter('lifetime',
                            fget=self._get_lifetime,
                            fset=do_nothing,
                            unit=q.h,
                            doc="Lifetime of the ring",
                            owner_only=True)]

        super(StorageRing, self).__init__(params)

    def _get_current(self):
        raise NotImplementedError

    def _get_energy(self):
        raise NotImplementedError

    def _get_lifetime(self):
        raise NotImplementedError
