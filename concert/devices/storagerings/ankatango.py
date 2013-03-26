"""
ANKA storage ring access via TANGO.
"""
from concert.devices.storagerings.base import StorageRing
import quantities as q


class ANKATangoStorageRing(StorageRing):
    def __init__(self, tango_connection):
        super(ANKATangoStorageRing, self).__init__()
        self._connection = tango_connection

    def _get_current(self):
        return self._connection.read_value("ECurrent") * q.mA

    def _get_energy(self):
        return self._connection.read_value("Energy") * q.MeV

    def _get_lifetime(self):
        return self._connection.read_value("Lifetime") * q.hour
