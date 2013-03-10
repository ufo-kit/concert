'''
Created on Mar 5, 2013

@author: farago
'''
from control.devices.storagerings.storagering import StorageRing
import quantities as pq


class ANKATangoStorageRing(StorageRing):
    def __init__(self, connection):
        super(ANKATangoStorageRing, self).__init__()
        self._connection = connection
        
    @property
    def current(self):
        return self._connection.tango_device.read_attribute("ECurrent").value*\
                                                                        pq.mA
    @property
    def energy(self):
        return self._connection.tango_device.read_attribute("Energy").value*\
                                                                        pq.GeV
    @property
    def lifetime(self):
        return self._connection.tango_device.read_attribute("Lifetime").value*\
                                                                        pq.hour