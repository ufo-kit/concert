'''
Created on Mar 5, 2013

@author: farago
'''
from concert.base import ConcertObject


class StorageRing(ConcertObject):
    """Electron storage ring."""
    @property
    def current(self):
        raise NotImplementedError

    @property
    def energy(self):
        raise NotImplementedError

    @property
    def lifetime(self):
        raise NotImplementedError
