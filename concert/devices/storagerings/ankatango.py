"""
ANKA storage ring access via TANGO.
"""
from concert.devices.storagerings.base import StorageRing
import quantities as q
from concert.connections.tango import TopoTomo


class StorageRing(StorageRing):
    def __init__(self):
        super(StorageRing, self).__init__()
        # TODO: find non-beam line specific storage ring device
        self._device = TopoTomo().get_device("iss/pvss/ANKAStatus")

    def _get_current(self):
        return self._device.ECurrent * q.mA

    def _get_energy(self):
        return self._device.Energy * q.MeV

    def _get_lifetime(self):
        return self._device.Lifetime * q.hour
