'''
Created on Mar 5, 2013

@author: farago
'''
from concert.devices.storagerings.base import StorageRing
import quantities as pq


class ANKATangoStorageRing(StorageRing):
    def __init__(self, connection):
        super(ANKATangoStorageRing, self).__init__()
        self._connection = connection

    def _get_tango_value(self, attrib):
        return self._connection.tango_device.read_attribute(attrib).value

    @property
    def current(self):
        return self._get_tango_value("ECurrent").value * pq.mA

    @property
    def energy(self):
        return self._get_tango_value("Energy").value * pq.MeV

    @property
    def lifetime(self):
        return self._get_tango_value("Lifetime").value * pq.hour
