'''
Created on Mar 5, 2013

@author: farago
'''
from control.controlobject import Identifiable


class StorageRing(Identifiable):
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
